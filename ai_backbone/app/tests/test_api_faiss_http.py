import httpx
import pytest

from app.core.config import Settings
from app.retrieval.faiss_http import FaissHTTPRetrievalProvider


@pytest.mark.asyncio
async def test_api_rag_index_and_retrieve_with_faiss_http(client, app_instance):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/index"):
            captured["index_headers"] = dict(request.headers)
            return httpx.Response(
                200,
                json={"collection": "sample-docs", "indexed_count": 1, "total_count": 1, "latency_ms": 12.3},
            )
        if request.url.path.endswith("/retrieve"):
            return httpx.Response(
                200,
                json={
                    "collection": "sample-docs",
                    "results": [
                        {
                            "id": "doc-1",
                            "text": "Value at Risk measures potential loss.",
                            "score": 0.9,
                            "metadata": {"category": "risk"},
                        }
                    ],
                    "latency_ms": 14.2,
                    "metadata": {},
                },
            )
        if request.url.path.endswith("/collections"):
            return httpx.Response(
                200,
                json={
                    "collections": [
                        {"name": "sample-docs", "document_count": 1, "index_exists": True, "metadata": {}},
                    ]
                },
            )
        return httpx.Response(404)

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890", faiss_http_enabled=True),
        transport=httpx.MockTransport(handler),
    )
    app_instance.state.container.retrieval_registry.register(provider)

    index_resp = await client.post(
        "/v1/rag/index",
        headers={"X-Request-ID": "faiss-api-1"},
        json={
            "provider": "faiss-http",
            "collection": "sample-docs",
            "documents": [{"id": "doc-1", "content": "Value at Risk measures potential loss.", "metadata": {}}],
            "options": {},
        },
    )
    assert index_resp.status_code == 200
    assert index_resp.json()["indexed_count"] == 1
    assert captured["index_headers"]["x-request-id"] == "faiss-api-1"

    retrieve_resp = await client.post(
        "/v1/rag/retrieve",
        json={
            "provider": "faiss-http",
            "collection": "sample-docs",
            "query": "What is VaR?",
            "top_k": 5,
            "filters": {},
            "options": {},
        },
    )
    assert retrieve_resp.status_code == 200
    body = retrieve_resp.json()
    assert body["chunks"][0]["id"] == "doc-1"
    assert body["metadata"]["provider"] == "faiss-http"


@pytest.mark.asyncio
async def test_api_collections_includes_faiss_worker_collections(client, app_instance):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/collections"):
            return httpx.Response(
                200,
                json={
                    "collections": [
                        {"name": "faiss-set", "document_count": 2, "index_exists": True, "metadata": {}},
                    ]
                },
            )
        return httpx.Response(404)

    provider = FaissHTTPRetrievalProvider(
        settings=Settings(faiss_worker_url="http://faiss:8890", faiss_http_enabled=True),
        transport=httpx.MockTransport(handler),
    )
    app_instance.state.container.retrieval_registry.register(provider)

    response = await client.get("/v1/collections")
    assert response.status_code == 200
    names = {item["name"] for item in response.json()["collections"]}
    assert "faiss-set" in names
