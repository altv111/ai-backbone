import json

import httpx
import pytest

from app.bootstrap import build_container
from app.contracts.llm import ChatMessage, ChatRequest, EmbedRequest
from app.core.config import Settings
from app.core.errors import (
    InvalidProviderResponseError,
    ProviderBusyError,
    ProviderFailedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedOperationError,
)
from app.core.request_context import reset_request_id, set_request_id
from app.providers.gemma import GemmaWorker, GemmaWorkerPool, LocalGemmaProvider


@pytest.mark.asyncio
async def test_local_gemma_registers_when_enabled():
    settings = Settings(
        enable_mock_providers=False,
        gemma_enabled=True,
        gemma_workers_json=[{"id": "gemma-01", "url": "http://worker:8888", "max_concurrent": 1}],
    )
    container = build_container(settings)
    assert container.llm_registry.exists("local-gemma")


@pytest.mark.asyncio
async def test_local_gemma_health_unavailable_when_no_workers():
    provider = LocalGemmaProvider(settings=Settings(gemma_workers_json=[]))
    health = await provider.health()
    assert health["status"] in {"degraded", "unavailable"}


@pytest.mark.asyncio
async def test_local_gemma_worker_max_concurrent_is_forced_to_one():
    provider = LocalGemmaProvider(
        settings=Settings(gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 5}])
    )
    assert provider._pool._workers[0].max_concurrent == 1


@pytest.mark.asyncio
async def test_worker_pool_selects_least_busy_healthy_worker():
    pool = GemmaWorkerPool(
        workers=[
            GemmaWorker(id="w1", url="http://w1", max_concurrent=2, active_requests=1, avg_latency_ms=100),
            GemmaWorker(id="w2", url="http://w2", max_concurrent=2, active_requests=0, avg_latency_ms=150),
        ]
    )
    selected = await pool.select_worker()
    assert selected.id == "w2"


@pytest.mark.asyncio
async def test_worker_pool_busy_returns_provider_busy():
    pool = GemmaWorkerPool(
        workers=[
            GemmaWorker(id="w1", url="http://w1", max_concurrent=1, active_requests=1),
            GemmaWorker(id="w2", url="http://w2", max_concurrent=1, active_requests=1),
        ]
    )
    with pytest.raises(ProviderBusyError):
        await pool.select_worker()


@pytest.mark.asyncio
async def test_worker_pool_no_healthy_workers_returns_provider_unavailable():
    pool = GemmaWorkerPool(
        workers=[
            GemmaWorker(id="w1", url="http://w1", max_concurrent=1, active_requests=0, healthy=False),
        ]
    )
    with pytest.raises(ProviderUnavailableError):
        await pool.select_worker()


@pytest.mark.asyncio
async def test_local_gemma_releases_worker_after_success_and_metadata_fields():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "content": "gemma-answer",
                "model_name": "gemma-12b-it",
                "latency_ms": 123,
                "tokens_generated": 88,
            },
        )

    settings = Settings(
        gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}],
        gemma_default_model="gemma-12b-it",
        gemma_default_temperature=0.2,
        gemma_default_max_new_tokens=512,
    )
    provider = LocalGemmaProvider(settings=settings, transport=httpx.MockTransport(handler))

    token = set_request_id("req-gemma")
    try:
        response = await provider.chat(
            ChatRequest(
                provider="local-gemma",
                messages=[
                    ChatMessage(role="system", content="You are helpful"),
                    ChatMessage(role="user", content="Explain VaR in simple terms"),
                ],
            )
        )
    finally:
        reset_request_id(token)

    payload = captured["json"]
    assert set(payload.keys()) == {
        "request_id",
        "model_name",
        "prompt",
        "temperature",
        "max_new_tokens",
    }
    assert payload["request_id"] == "req-gemma"
    assert payload["model_name"] == "gemma-12b-it"
    assert payload["temperature"] == 0.2
    assert payload["max_new_tokens"] == 512
    assert payload["prompt"] == "Explain VaR in simple terms"

    assert response.metadata.extra["worker_id"] == "w1"
    assert response.metadata.extra["worker_latency_ms"] == 123
    assert response.metadata.extra["tokens_generated"] == 88
    assert provider._pool._workers[0].active_requests == 0


@pytest.mark.asyncio
async def test_local_gemma_releases_worker_after_failure():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="worker-error")

    settings = Settings(gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}])
    provider = LocalGemmaProvider(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(ProviderFailedError):
        await provider.chat(
            ChatRequest(
                provider="local-gemma",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )

    assert provider._pool._workers[0].active_requests == 0


@pytest.mark.asyncio
async def test_local_gemma_missing_content_maps_to_invalid_provider_response():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model_name": "gemma-12b-it"})

    settings = Settings(gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}])
    provider = LocalGemmaProvider(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(InvalidProviderResponseError):
        await provider.chat(
            ChatRequest(provider="local-gemma", messages=[ChatMessage(role="user", content="hello")])
        )


@pytest.mark.asyncio
async def test_local_gemma_timeout_maps_to_provider_timeout():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    settings = Settings(gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}])
    provider = LocalGemmaProvider(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(ProviderTimeoutError):
        await provider.chat(
            ChatRequest(provider="local-gemma", messages=[ChatMessage(role="user", content="hello")])
        )


@pytest.mark.asyncio
async def test_local_gemma_embed_unsupported_operation():
    settings = Settings(gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}])
    provider = LocalGemmaProvider(settings=settings)

    with pytest.raises(UnsupportedOperationError):
        await provider.embed(EmbedRequest(provider="local-gemma", texts=["hello"]))
