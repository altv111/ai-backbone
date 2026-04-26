import pathlib
import sys

import httpx
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worker_services.faiss_worker.app import create_app
from worker_services.faiss_worker.config import FaissWorkerSettings
from worker_services.faiss_worker.embedding import MockEmbeddingService
from worker_services.faiss_worker.index_store import FaissIndexStore, MockIndexStore


async def _get_client(app):
    await app.router._startup()
    state = app.state.worker_state
    if state.settings.mock_mode:
        state.store = MockIndexStore()
    else:
        state.store = FaissIndexStore(
            index_root=state.settings.index_root,
            embedding_model_name=state.settings.embedding_model,
            embedding_service=MockEmbeddingService(),
        )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://faiss-worker")


@pytest.mark.asyncio
async def test_faiss_worker_health_works_in_mock_mode():
    app = create_app(FaissWorkerSettings(mock_mode=True))
    client = await _get_client(app)
    try:
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["mock_mode"] is True
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_faiss_worker_index_append_and_replace_and_duplicate_replace():
    app = create_app(FaissWorkerSettings(mock_mode=True))
    client = await _get_client(app)
    try:
        response1 = await client.post(
            "/index",
            json={
                "collection": "sample",
                "documents": [{"id": "doc-1", "text": "alpha text", "metadata": {"project": "MRAC"}}],
                "mode": "append",
            },
        )
        assert response1.status_code == 200

        response2 = await client.post(
            "/index",
            json={
                "collection": "sample",
                "documents": [{"id": "doc-1", "text": "alpha text updated", "metadata": {"project": "MRAC"}}],
                "mode": "append",
            },
        )
        assert response2.status_code == 200
        assert response2.json()["total_count"] == 1

        response3 = await client.post(
            "/index",
            json={
                "collection": "sample",
                "documents": [{"id": "doc-2", "text": "beta text", "metadata": {}}],
                "mode": "replace",
            },
        )
        assert response3.status_code == 200
        assert response3.json()["total_count"] == 1
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_faiss_worker_retrieve_top_k_filters_and_missing_collection():
    app = create_app(FaissWorkerSettings(mock_mode=True))
    client = await _get_client(app)
    try:
        await client.post(
            "/index",
            json={
                "collection": "sample",
                "documents": [
                    {"id": "doc-1", "text": "var risk explanation", "metadata": {"project": "MRAC"}},
                    {"id": "doc-2", "text": "confluence integration", "metadata": {"project": "ABC"}},
                ],
                "mode": "append",
            },
        )

        retrieve = await client.post(
            "/retrieve",
            json={
                "collection": "sample",
                "query": "var",
                "top_k": 1,
                "filters": {"project": "MRAC"},
            },
        )
        assert retrieve.status_code == 200
        body = retrieve.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["id"] == "doc-1"

        missing = await client.post(
            "/retrieve",
            json={"collection": "missing", "query": "x", "top_k": 1, "filters": {}},
        )
        assert missing.status_code == 404
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_faiss_worker_collections_and_validation_errors():
    app = create_app(FaissWorkerSettings(mock_mode=True, max_top_k=50))
    client = await _get_client(app)
    try:
        empty_docs = await client.post(
            "/index",
            json={"collection": "sample", "documents": [], "mode": "append"},
        )
        assert empty_docs.status_code == 400

        bad_topk = await client.post(
            "/retrieve",
            json={"collection": "sample", "query": "x", "top_k": 0, "filters": {}},
        )
        assert bad_topk.status_code == 400

        await client.post(
            "/index",
            json={
                "collection": "sample",
                "documents": [{"id": "doc-1", "text": "hello", "metadata": {}}],
                "mode": "append",
            },
        )
        collections = await client.get("/collections")
        assert collections.status_code == 200
        assert len(collections.json()["collections"]) >= 1
    finally:
        await client.aclose()
        await app.router._shutdown()


@pytest.mark.asyncio
async def test_faiss_worker_starts_in_mock_mode_without_faiss():
    app = create_app(FaissWorkerSettings(mock_mode=True))
    client = await _get_client(app)
    try:
        response = await client.get("/health")
        assert response.status_code == 200
    finally:
        await client.aclose()
        await app.router._shutdown()
