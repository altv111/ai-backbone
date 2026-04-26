import json
from typing import Any

import httpx

from app.contracts.common import ResponseMetadata
from app.contracts.llm import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse
from app.contracts.providers import ProviderMetadata
from app.core.config import Settings
from app.core.errors import (
    InvalidProviderResponseError,
    InvalidRequestError,
    ProviderFailedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedOperationError,
)
from app.core.request_context import get_request_id
from app.providers._formatting import messages_to_text
from app.providers.base import LLMProvider


class CentralLLMProvider(LLMProvider):
    name = "central-llm"
    description = "Organization central LLM API"
    supports_chat = True
    supports_embeddings = False

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    async def chat(self, request: ChatRequest) -> ChatResponse:
        email = request.options.get("email")
        if not email:
            raise InvalidRequestError("Missing required option: email")

        kannon_id = request.options.get("kannon_id")
        if not kannon_id:
            raise InvalidRequestError("Missing required option: kannon_id")

        api_key = request.options.get("api_key") or self._settings.central_llm_api_key
        if not api_key:
            raise ProviderUnavailableError(
                "Provider 'central-llm' API key is unavailable",
                {"missing": ["api_key"], "provider": self.name},
            )
        if not self._settings.central_llm_api_url:
            raise ProviderUnavailableError(
                "Provider 'central-llm' is not fully configured",
                {"missing": ["CENTRAL_LLM_API_URL"], "provider": self.name},
            )

        model_name = request.model or self._settings.central_llm_default_model
        if model_name not in self._settings.central_llm_allowed_models:
            raise InvalidRequestError(
                f"Model '{model_name}' is not allowed for provider '{self.name}'",
                {
                    "provider": self.name,
                    "allowed_models": self._settings.central_llm_allowed_models,
                },
            )

        payload = {
            "email": email,
            "apikey": api_key,
            "data_classification": request.options.get("data_classification")
            or self._settings.central_llm_default_data_classification,
            "message": messages_to_text(request.messages),
            "kannon_id": kannon_id,
            "model_name": model_name,
        }

        headers = {"Content-Type": "application/json"}
        request_id = get_request_id()
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.central_llm_timeout_seconds,
                verify=self._settings.central_llm_verify_ssl,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self._settings.central_llm_api_url,
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Provider 'central-llm' timed out",
                {"provider": self.name},
            ) from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderFailedError(
                f"Provider '{self.name}' returned status {response.status_code}",
                {
                    "provider": self.name,
                    "status_code": response.status_code,
                    "response": _sanitize_text(response.text),
                },
            )

        content = _extract_response_content(response)
        if not content.strip():
            raise InvalidProviderResponseError(
                f"Provider '{self.name}' returned empty content",
                {"provider": self.name},
            )

        return ChatResponse(
            content=content,
            metadata=ResponseMetadata(
                request_id=request_id,
                provider=self.name,
                model=model_name,
                extra={"status_code": response.status_code},
            ),
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        raise UnsupportedOperationError(f"Provider '{self.name}' does not support embeddings")

    async def health(self) -> dict[str, Any]:
        missing = []
        if not self._settings.central_llm_api_url:
            missing.append("CENTRAL_LLM_API_URL")
        if not self._settings.central_llm_api_key:
            missing.append("CENTRAL_LLM_API_KEY")
        status = "ok" if not missing else "degraded"
        return {
            "status": status,
            "provider": self.name,
            "missing": missing,
        }

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            type="llm",
            description=self.description,
            capabilities=["chat"],
            metadata={
                "default_model": self._settings.central_llm_default_model,
                "allowed_models": self._settings.central_llm_allowed_models,
            },
        )


def _extract_response_content(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return response.text

    if isinstance(payload, dict):
        for key in ("content", "response", "message", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
    return response.text


def _sanitize_text(text: str, limit: int = 500) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
