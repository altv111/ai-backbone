import pytest


@pytest.mark.asyncio
async def test_rag_index_with_mock_provider(client):
    response = await client.post(
        "/v1/rag/index",
        json={
            "provider": "mock-rag",
            "collection": "team-docs",
            "documents": [
                {"id": "1", "content": "alpha document"},
                {"id": "2", "content": "beta document"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["indexed_count"] == 2


@pytest.mark.asyncio
async def test_rag_retrieve_with_mock_provider(client):
    await client.post(
        "/v1/rag/index",
        json={
            "provider": "mock-rag",
            "collection": "retrieval-test",
            "documents": [
                {"id": "1", "content": "jira feature planning"},
                {"id": "2", "content": "confluence architecture notes"},
            ],
        },
    )

    response = await client.post(
        "/v1/rag/retrieve",
        json={
            "provider": "mock-rag",
            "collection": "retrieval-test",
            "query": "jira",
            "top_k": 1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == 1
    assert body["chunks"][0]["id"] == "1"


@pytest.mark.asyncio
async def test_rag_top_k_limit_returns_invalid_request(client):
    response = await client.post(
        "/v1/rag/retrieve",
        json={
            "provider": "mock-rag",
            "collection": "default",
            "query": "anything",
            "top_k": 100,
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
