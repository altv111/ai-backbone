import pytest
import httpx

from app.core.config import Settings
from app.providers.gemma import LocalGemmaProvider


@pytest.mark.asyncio
async def test_llm_chat_with_mock_provider(client):
    response = await client.post(
        "/v1/llm/chat",
        json={
            "provider": "mock-llm",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "hello" in body["content"]
    assert body["metadata"]["provider"] == "mock-llm"


@pytest.mark.asyncio
async def test_llm_embed_with_mock_provider(client):
    response = await client.post(
        "/v1/llm/embed",
        json={
            "provider": "mock-llm",
            "texts": ["alpha", "beta"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 8


@pytest.mark.asyncio
async def test_unknown_llm_provider_returns_structured_error(client):
    response = await client.post(
        "/v1/llm/chat",
        json={
            "provider": "does-not-exist",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "provider_not_found"


@pytest.mark.asyncio
async def test_message_limit_returns_invalid_request(client):
    response = await client.post(
        "/v1/llm/chat",
        json={
            "provider": "mock-llm",
            "messages": [{"role": "user", "content": str(i)} for i in range(51)],
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_chat_audit_includes_email_and_ask(client, app_instance):
    events = []
    app_instance.state.container.audit_service.emit = lambda event: events.append(event)

    response = await client.post(
        "/v1/llm/chat",
        json={
            "provider": "mock-llm",
            "messages": [{"role": "user", "content": "hello audit"}],
            "options": {"email": "dev@example.com"},
        },
        headers={"X-Request-ID": "audit-req-1"},
    )
    assert response.status_code == 200
    assert len(events) >= 1
    audit = events[-1]
    assert audit["event"] == "llm_chat"
    assert audit["request_id"] == "audit-req-1"
    assert audit["email"] == "dev@example.com"
    assert audit["ask"] == "hello audit"
    assert audit["status"] == "success"


@pytest.mark.asyncio
async def test_chat_audit_includes_wait_time_for_local_gemma(client, app_instance):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": "ok",
                "model_name": "gemma-12b-it",
                "latency_ms": 25.0,
                "tokens_generated": 3,
            },
        )

    provider = LocalGemmaProvider(
        settings=Settings(
            gemma_enabled=True,
            gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}],
        ),
        transport=httpx.MockTransport(handler),
    )
    app_instance.state.container.llm_registry.register(provider)

    events = []
    app_instance.state.container.audit_service.emit = lambda event: events.append(event)

    response = await client.post(
        "/v1/llm/chat",
        json={
            "provider": "local-gemma",
            "messages": [{"role": "user", "content": "what is var"}],
            "options": {"email": "ops@example.com"},
        },
    )
    assert response.status_code == 200
    audit = events[-1]
    assert audit["provider"] == "local-gemma"
    assert audit["email"] == "ops@example.com"
    assert isinstance(audit["wait_time_ms"], (int, float))
