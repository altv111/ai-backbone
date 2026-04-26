import time
from datetime import datetime, timezone
from collections.abc import AsyncIterator

from app.contracts.llm import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse
from app.core.audit import AuditService
from app.core.config import Settings
from app.core.errors import (
    AppError,
    InvalidRequestError,
    ProviderFailedError,
    UnsupportedOperationError,
)
from app.core.registry import LLMProviderRegistry
from app.core.request_context import get_request_id


class LLMService:
    def __init__(self, registry: LLMProviderRegistry, settings: Settings, audit_service: AuditService) -> None:
        self.registry = registry
        self.settings = settings
        self.audit_service = audit_service

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if len(request.messages) > self.settings.max_messages:
            raise InvalidRequestError(
                "Too many messages in request",
                {"max_messages": self.settings.max_messages},
            )

        provider = self.registry.get(request.provider)
        if not provider.supports_chat:
            raise UnsupportedOperationError(f"Provider '{provider.name}' does not support chat")

        start = time.perf_counter()
        try:
            response = await provider.chat(request)
        except AppError as exc:
            self._emit_chat_audit(
                request=request,
                provider=provider.name,
                status="error",
                error_code=exc.code,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
                model=request.model,
            )
            raise
        except Exception as exc:
            wrapped = ProviderFailedError(f"Provider '{provider.name}' failed while handling chat")
            self._emit_chat_audit(
                request=request,
                provider=provider.name,
                status="error",
                error_code=wrapped.code,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
                model=request.model,
            )
            raise wrapped from exc
        response.metadata.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        self._emit_chat_audit(
            request=request,
            provider=provider.name,
            status="success",
            error_code=None,
            latency_ms=response.metadata.latency_ms,
            model=response.metadata.model or request.model,
            wait_time_ms=response.metadata.extra.get("queue_wait_ms"),
        )
        return response

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        if len(request.texts) > self.settings.max_texts_per_embed:
            raise InvalidRequestError(
                "Too many texts in embedding request",
                {"max_texts_per_embed": self.settings.max_texts_per_embed},
            )

        provider = self.registry.get(request.provider)
        if not provider.supports_embeddings:
            raise UnsupportedOperationError(f"Provider '{provider.name}' does not support embeddings")

        start = time.perf_counter()
        try:
            response = await provider.embed(request)
        except AppError:
            raise
        except Exception as exc:
            raise ProviderFailedError(f"Provider '{provider.name}' failed while handling embeddings") from exc
        response.metadata.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return response

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        if len(request.messages) > self.settings.max_messages:
            raise InvalidRequestError(
                "Too many messages in request",
                {"max_messages": self.settings.max_messages},
            )

        provider = self.registry.get(request.provider)
        if not getattr(provider, "supports_streaming", False):
            raise UnsupportedOperationError(f"Provider '{provider.name}' does not support streaming")

        try:
            return await provider.chat_stream(request)
        except AppError:
            raise
        except Exception as exc:
            raise ProviderFailedError(f"Provider '{provider.name}' failed while handling stream chat") from exc

    def _emit_chat_audit(
        self,
        request: ChatRequest,
        provider: str,
        status: str,
        error_code: str | None,
        latency_ms: float | None,
        model: str | None,
        wait_time_ms: float | None = None,
    ) -> None:
        ask = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
        if len(ask) > 500:
            ask = ask[:500]

        self.audit_service.emit(
            {
                "event": "llm_chat",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": get_request_id(),
                "provider": provider,
                "email": request.options.get("email"),
                "ask": ask,
                "wait_time_ms": wait_time_ms,
                "latency_ms": latency_ms,
                "status": status,
                "error_code": error_code,
                "model": model,
            }
        )
