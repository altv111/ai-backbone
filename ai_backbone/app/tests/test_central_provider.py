import json

import httpx
import pytest

from app.bootstrap import build_container
from app.contracts.llm import ChatMessage, ChatRequest, EmbedRequest
from app.core.config import Settings
from app.core.errors import (
    InvalidRequestError,
    ProviderFailedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedOperationError,
)
from app.core.request_context import reset_request_id, set_request_id
from app.providers.central_llm import CentralLLMProvider


@pytest.mark.asyncio
async def test_central_registers_when_enabled():
    settings = Settings(
        enable_mock_providers=False,
        central_llm_enabled=True,
        central_llm_api_url="https://central.example/api",
        central_llm_api_key="k",
    )
    container = build_container(settings)
    assert container.llm_registry.exists("central-llm")


@pytest.mark.asyncio
async def test_central_not_registered_when_disabled():
    settings = Settings(enable_mock_providers=False, central_llm_enabled=False)
    container = build_container(settings)
    assert not container.llm_registry.exists("central-llm")


@pytest.mark.asyncio
async def test_central_chat_sends_exact_payload_keys_and_mappings():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"content": "ok"})

    settings = Settings(
        central_llm_enabled=True,
        central_llm_api_url="https://central.example/api",
        central_llm_api_key="settings-key",
        central_llm_allowed_models=["gemini-flash", "gemini-pro"],
    )
    provider = CentralLLMProvider(settings=settings, transport=httpx.MockTransport(handler))

    token = set_request_id("req-123")
    try:
        response = await provider.chat(
            ChatRequest(
                provider="central-llm",
                model="gemini-pro",
                messages=[ChatMessage(role="user", content="hello")],
                options={"email": "dev@example.com", "kannon_id": "k-1"},
            )
        )
    finally:
        reset_request_id(token)

    payload = captured["json"]
    assert set(payload.keys()) == {
        "email",
        "apikey",
        "data_classification",
        "message",
        "kannon_id",
        "model_name",
    }
    assert payload["email"] == "dev@example.com"
    assert payload["apikey"] == "settings-key"
    assert payload["kannon_id"] == "k-1"
    assert payload["model_name"] == "gemini-pro"
    assert captured["headers"]["x-request-id"] == "req-123"
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_central_uses_default_model_when_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model_name"] == "gemini-flash"
        return httpx.Response(200, json={"content": "ok"})

    settings = Settings(
        central_llm_api_url="https://central.example/api",
        central_llm_api_key="settings-key",
    )
    provider = CentralLLMProvider(settings=settings, transport=httpx.MockTransport(handler))

    await provider.chat(
        ChatRequest(
            provider="central-llm",
            messages=[ChatMessage(role="user", content="hello")],
            options={"email": "dev@example.com", "kannon_id": "k-1"},
        )
    )


@pytest.mark.asyncio
async def test_central_missing_email_invalid_request():
    settings = Settings(central_llm_api_url="https://central.example/api", central_llm_api_key="settings-key")
    provider = CentralLLMProvider(settings=settings)

    with pytest.raises(InvalidRequestError):
        await provider.chat(
            ChatRequest(
                provider="central-llm",
                messages=[ChatMessage(role="user", content="hello")],
                options={"kannon_id": "k-1"},
            )
        )


@pytest.mark.asyncio
async def test_central_missing_kannon_id_invalid_request():
    settings = Settings(central_llm_api_url="https://central.example/api", central_llm_api_key="settings-key")
    provider = CentralLLMProvider(settings=settings)

    with pytest.raises(InvalidRequestError):
        await provider.chat(
            ChatRequest(
                provider="central-llm",
                messages=[ChatMessage(role="user", content="hello")],
                options={"email": "dev@example.com"},
            )
        )


@pytest.mark.asyncio
async def test_central_missing_api_key_provider_unavailable():
    settings = Settings(central_llm_api_url="https://central.example/api", central_llm_api_key="")
    provider = CentralLLMProvider(settings=settings)

    with pytest.raises(ProviderUnavailableError):
        await provider.chat(
            ChatRequest(
                provider="central-llm",
                messages=[ChatMessage(role="user", content="hello")],
                options={"email": "dev@example.com", "kannon_id": "k-1"},
            )
        )


@pytest.mark.asyncio
async def test_central_invalid_model_invalid_request():
    settings = Settings(
        central_llm_api_url="https://central.example/api",
        central_llm_api_key="settings-key",
        central_llm_allowed_models=["gemini-flash"],
    )
    provider = CentralLLMProvider(settings=settings)

    with pytest.raises(InvalidRequestError):
        await provider.chat(
            ChatRequest(
                provider="central-llm",
                model="unknown-model",
                messages=[ChatMessage(role="user", content="hello")],
                options={"email": "dev@example.com", "kannon_id": "k-1"},
            )
        )


@pytest.mark.asyncio
async def test_central_embed_unsupported_operation():
    settings = Settings()
    provider = CentralLLMProvider(settings=settings)

    with pytest.raises(UnsupportedOperationError):
        await provider.embed(EmbedRequest(provider="central-llm", texts=["hello"]))


@pytest.mark.asyncio
async def test_central_parses_raw_text_response():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="plain-text-response")

    settings = Settings(central_llm_api_url="https://central.example/api", central_llm_api_key="settings-key")
    provider = CentralLLMProvider(settings=settings, transport=httpx.MockTransport(handler))

    response = await provider.chat(
        ChatRequest(
            provider="central-llm",
            messages=[ChatMessage(role="user", content="hello")],
            options={"email": "dev@example.com", "kannon_id": "k-1"},
        )
    )
    assert response.content == "plain-text-response"


@pytest.mark.asyncio
async def test_central_parses_json_response():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "json-response"})

    settings = Settings(central_llm_api_url="https://central.example/api", central_llm_api_key="settings-key")
    provider = CentralLLMProvider(settings=settings, transport=httpx.MockTransport(handler))

    response = await provider.chat(
        ChatRequest(
            provider="central-llm",
            messages=[ChatMessage(role="user", content="hello")],
            options={"email": "dev@example.com", "kannon_id": "k-1"},
        )
    )
    assert response.content == "json-response"


@pytest.mark.asyncio
async def test_central_non_2xx_maps_to_provider_failed():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream-error")

    settings = Settings(central_llm_api_url="https://central.example/api", central_llm_api_key="settings-key")
    provider = CentralLLMProvider(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(ProviderFailedError):
        await provider.chat(
            ChatRequest(
                provider="central-llm",
                messages=[ChatMessage(role="user", content="hello")],
                options={"email": "dev@example.com", "kannon_id": "k-1"},
            )
        )


@pytest.mark.asyncio
async def test_central_timeout_maps_to_provider_timeout():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    settings = Settings(central_llm_api_url="https://central.example/api", central_llm_api_key="settings-key")
    provider = CentralLLMProvider(settings=settings, transport=httpx.MockTransport(handler))

    with pytest.raises(ProviderTimeoutError):
        await provider.chat(
            ChatRequest(
                provider="central-llm",
                messages=[ChatMessage(role="user", content="hello")],
                options={"email": "dev@example.com", "kannon_id": "k-1"},
            )
        )
