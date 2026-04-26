import pytest


@pytest.mark.asyncio
async def test_knowledge_answer_with_mock_provider(client):
    response = await client.post(
        "/v1/knowledge/answer",
        json={
            "provider": "mock-knowledge",
            "query": "What is Layer 1?",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "Layer 1" in body["answer"]
    assert len(body["sources"]) >= 1


@pytest.mark.asyncio
async def test_knowledge_search_with_mock_provider(client):
    response = await client.post(
        "/v1/knowledge/search",
        json={
            "provider": "mock-knowledge",
            "query": "OMNI",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 3
