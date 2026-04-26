import httpx
import pytest

from app.core.config import Settings
from app.providers.central_llm import CentralLLMProvider
from app.providers.gemma import LocalGemmaProvider


@pytest.mark.asyncio
async def test_api_chat_central_llm_works_with_mocked_http(client, app_instance):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": "central-ok"})

    provider = CentralLLMProvider(
        settings=Settings(
            central_llm_enabled=True,
            central_llm_api_url="https://central.example/api",
            central_llm_api_key="settings-key",
        ),
        transport=httpx.MockTransport(handler),
    )
    app_instance.state.container.llm_registry.register(provider)

    response = await client.post(
        "/v1/llm/chat",
        json={
            "provider": "central-llm",
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"email": "dev@example.com", "kannon_id": "k-1"},
        },
        headers={"X-Request-ID": "api-central-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "central-ok"
    assert body["metadata"]["provider"] == "central-llm"


@pytest.mark.asyncio
async def test_api_chat_local_gemma_works_with_mocked_http(client, app_instance):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": "gemma-ok", "worker_id": "w1"})

    provider = LocalGemmaProvider(
        settings=Settings(
            gemma_enabled=True,
            gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}],
        ),
        transport=httpx.MockTransport(handler),
    )
    app_instance.state.container.llm_registry.register(provider)

    response = await client.post(
        "/v1/llm/chat",
        json={
            "provider": "local-gemma",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "gemma-ok"
    assert body["metadata"]["provider"] == "local-gemma"


@pytest.mark.asyncio
async def test_health_providers_endpoint_works(client):
    response = await client.get("/v1/health/providers")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm" in body["providers"]


@pytest.mark.asyncio
async def test_provider_busy_structured_error_with_request_id(client, app_instance):
    provider = LocalGemmaProvider(
        settings=Settings(gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}])
    )
    provider._pool._workers[0].active_requests = 1
    app_instance.state.container.llm_registry.register(provider)

    response = await client.post(
        "/v1/llm/chat",
        json={"provider": "local-gemma", "messages": [{"role": "user", "content": "hello"}]},
        headers={"X-Request-ID": "req-busy-1"},
    )
    assert response.status_code == 429
    body = response.json()
    assert body["error"]["code"] == "provider_busy"
    assert body["error"]["request_id"] == "req-busy-1"


@pytest.mark.asyncio
async def test_unsupported_embedding_call_structured_error(client, app_instance):
    provider = LocalGemmaProvider(
        settings=Settings(gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}])
    )
    app_instance.state.container.llm_registry.register(provider)

    response = await client.post(
        "/v1/llm/embed",
        json={"provider": "local-gemma", "texts": ["a"]},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_operation"


@pytest.mark.asyncio
async def test_api_chat_stream_local_gemma_works(client, app_instance):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/generate/stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='event: delta\ndata: {"content":"hello from worker"}\n\n'
                'event: done\ndata: {"model_name":"gemma-12b-it","latency_ms":12.0,"tokens_generated":3}\n\n',
            )
        return httpx.Response(404, text="not found")

    provider = LocalGemmaProvider(
        settings=Settings(
            gemma_enabled=True,
            gemma_workers_json=[{"id": "w1", "url": "http://w1:8888", "max_concurrent": 1}],
        ),
        transport=httpx.MockTransport(handler),
    )
    app_instance.state.container.llm_registry.register(provider)

    response = await client.post(
        "/v1/llm/chat/stream",
        json={
            "provider": "local-gemma",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: delta" in response.text
    assert "hello from worker" in response.text


@pytest.mark.asyncio
async def test_api_chat_stream_non_local_provider_unsupported(client):
    response = await client.post(
        "/v1/llm/chat/stream",
        json={
            "provider": "mock-llm",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_operation"
