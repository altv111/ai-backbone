import json

import httpx
import pytest

from app.bootstrap import build_container
from app.contracts.rag import IndexRequest, RetrieveRequest
from app.core.config import Settings
from app.core.errors import (
    InvalidProviderResponseError,
    ProviderFailedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.request_context import reset_request_id, set_request_id
from app.retrieval.faiss_http import FaissHTTPRetrievalProvider


@pytest.mark.asyncio
async def test_faiss_http_registers_when_enabled():
    container = build_container(Settings(enable_mock_providers=False, faiss_http_enabled=True))
    assert container.retrieval_registry.exists("faiss-http")


@pytest.mark.asyncio
async def test_faiss_http_not_registered_when_disabled():
    container = build_container(Settings(enable_mock_providers=False, faiss_http_enabled=False))
    assert not container.retrieval_registry.exists("faiss-http")


@pytest.mark.asyncio
async def test_faiss_http_index_maps_request_and_forwards_request_id():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"collection": "sample-docs", "indexed_count": 1, "total_count": 1, "latency_ms": 10.0},
        )

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890", faiss_http_timeout_seconds=30),
        transport=httpx.MockTransport(handler),
    )

    token = set_request_id("req-faiss-1")
    try:
        response = await provider.index(
            IndexRequest(
                provider="faiss-http",
                collection="sample-docs",
                documents=[{"id": "doc-1", "content": "Value at Risk", "metadata": {"source": "manual"}}],
            )
        )
    finally:
        reset_request_id(token)

    assert captured["path"] == "/index"
    assert captured["headers"]["x-request-id"] == "req-faiss-1"
    assert captured["json"]["collection"] == "sample-docs"
    assert captured["json"]["documents"][0]["text"] == "Value at Risk"
    assert response.collection == "sample-docs"
    assert response.indexed_count == 1
    assert response.metadata.provider == "faiss-http"
    assert response.metadata.extra["worker_url"] == "http://faiss:8890"


@pytest.mark.asyncio
async def test_faiss_http_retrieve_maps_request_and_results():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "collection": "sample-docs",
                "results": [
                    {
                        "id": "doc-1",
                        "text": "Value at Risk measures potential loss.",
                        "score": 0.88,
                        "metadata": {"category": "risk"},
                    }
                ],
                "latency_ms": 15.0,
                "metadata": {},
            },
        )

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890"),
        transport=httpx.MockTransport(handler),
    )

    response = await provider.retrieve(
        RetrieveRequest(
            provider="faiss-http",
            collection="sample-docs",
            query="What is VaR?",
            top_k=5,
            filters={"category": "risk"},
        )
    )

    assert captured["path"] == "/retrieve"
    assert captured["json"]["query"] == "What is VaR?"
    assert captured["json"]["filters"] == {"category": "risk"}
    assert len(response.chunks) == 1
    assert response.chunks[0].id == "doc-1"
    assert response.chunks[0].content.startswith("Value at Risk")
    assert response.metadata.extra["collection"] == "sample-docs"


@pytest.mark.asyncio
async def test_faiss_http_timeout_maps_to_provider_timeout():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderTimeoutError):
        await provider.retrieve(
            RetrieveRequest(provider="faiss-http", collection="sample", query="q", top_k=1, filters={})
        )


@pytest.mark.asyncio
async def test_faiss_http_non_2xx_maps_to_provider_failed():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="worker failed")

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderFailedError):
        await provider.index(
            IndexRequest(provider="faiss-http", collection="sample", documents=[{"id": "a", "content": "x"}])
        )


@pytest.mark.asyncio
async def test_faiss_http_invalid_response_maps_to_invalid_provider_response():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidProviderResponseError):
        await provider.retrieve(
            RetrieveRequest(provider="faiss-http", collection="sample", query="q", top_k=1, filters={})
        )


@pytest.mark.asyncio
async def test_faiss_http_missing_worker_url_unavailable():
    provider = FaissHTTPRetrievalProvider(settings=Settings(faiss_worker_url=""))

    with pytest.raises(ProviderUnavailableError):
        await provider.index(
            IndexRequest(provider="faiss-http", collection="sample", documents=[{"id": "a", "content": "x"}])
        )


@pytest.mark.asyncio
async def test_faiss_http_health_unavailable_when_worker_url_missing():
    provider = FaissHTTPRetrievalProvider(settings=Settings(faiss_worker_url=""))
    health = await provider.health()
    assert health["status"] in {"unavailable", "degraded"}


@pytest.mark.asyncio
async def test_faiss_http_health_ok_when_worker_health_works():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890"),
        transport=httpx.MockTransport(handler),
    )
    health = await provider.health()
    assert health["status"] == "ok"
