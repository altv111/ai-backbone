import pathlib
import sys

import httpx
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worker_services.gemma_worker.app import create_app
from worker_services.gemma_worker.config import WorkerSettings
from worker_services.gemma_worker.model_loader import MockModelRunner, TransformersGemmaRunner


async def _get_client(app):
    await app.router._startup()
    state = app.state.worker_state
    if state.settings.mock_mode:
        state.runner = MockModelRunner()
    else:
        state.runner = TransformersGemmaRunner(
            model_name=state.settings.model_name,
            model_path=state.settings.model_path,
            num_threads=state.settings.num_threads,
            num_interop_threads=state.settings.num_interop_threads,
        )
    state.loaded = state.runner is not None
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://worker")
    return client


@pytest.mark.asyncio
async def test_worker_health_in_mock_mode():
    app = create_app(WorkerSettings(mock_mode=True, max_concurrent=1))
    client = await _get_client(app)
    try:
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["loaded"] is True
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_worker_generate_in_mock_mode_and_shape():
    app = create_app(WorkerSettings(mock_mode=True, max_concurrent=1))
    client = await _get_client(app)
    try:
        response = await client.post(
            "/generate",
            json={
                "request_id": "r1",
                "model_name": "gemma-12b-it",
                "prompt": "Explain VaR",
                "temperature": 0.2,
                "max_new_tokens": 64,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"content", "model_name", "latency_ms", "tokens_generated"}
        assert body["content"]
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_worker_max_concurrent_can_return_429_when_busy():
    app = create_app(WorkerSettings(mock_mode=True, max_concurrent=1))
    client = await _get_client(app)
    try:
        state = app.state.worker_state
        await state.semaphore.acquire()
        state.active_requests = state.settings.max_concurrent

        response = await client.post(
            "/generate",
            json={
                "request_id": "r1",
                "model_name": "gemma-12b-it",
                "prompt": "Explain VaR",
                "temperature": 0.2,
                "max_new_tokens": 64,
            },
        )
        assert response.status_code == 429
    finally:
        state = app.state.worker_state
        if state.semaphore.locked():
            state.semaphore.release()
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_worker_app_starts_in_mock_mode_without_real_model_load():
    app = create_app(WorkerSettings(mock_mode=True, max_concurrent=1))
    client = await _get_client(app)
    try:
        response = await client.get("/health")
        assert response.status_code == 200
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_worker_generate_stream_in_mock_mode():
    app = create_app(WorkerSettings(mock_mode=True, max_concurrent=1))
    client = await _get_client(app)
    try:
        response = await client.post(
            "/generate/stream",
            json={
                "request_id": "r1",
                "model_name": "gemma-12b-it",
                "prompt": "Explain VaR",
                "temperature": 0.2,
                "max_new_tokens": 64,
            },
        )
        assert response.status_code == 200
        assert "event: delta" in response.text
        assert "event: done" in response.text
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_worker_generate_stream_returns_429_when_busy():
    app = create_app(WorkerSettings(mock_mode=True, max_concurrent=1))
    client = await _get_client(app)
    try:
        state = app.state.worker_state
        await state.semaphore.acquire()
        state.active_requests = state.settings.max_concurrent
        response = await client.post(
            "/generate/stream",
            json={
                "request_id": "r1",
                "model_name": "gemma-12b-it",
                "prompt": "Explain VaR",
                "temperature": 0.2,
                "max_new_tokens": 64,
            },
        )
        assert response.status_code == 429
    finally:
        state = app.state.worker_state
        if state.semaphore.locked():
            state.semaphore.release()
        await client.aclose()
        await app.router._shutdown()


def test_worker_settings_force_max_concurrent_to_one():
    settings = WorkerSettings(max_concurrent=8)
    assert settings.max_concurrent == 1
