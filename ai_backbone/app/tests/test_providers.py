import pytest


@pytest.mark.asyncio
async def test_list_providers_contains_mock_providers(client):
    response = await client.get("/v1/providers")
    assert response.status_code == 200

    body = response.json()
    llm_names = {p["name"] for p in body["llm"]}
    retrieval_names = {p["name"] for p in body["retrieval"]}
    knowledge_names = {p["name"] for p in body["knowledge"]}

    assert "mock-llm" in llm_names
    assert "mock-rag" in retrieval_names
    assert "mock-knowledge" in knowledge_names


@pytest.mark.asyncio
async def test_collections_endpoint(client):
    response = await client.get("/v1/collections")
    assert response.status_code == 200
    body = response.json()
    assert "collections" in body
