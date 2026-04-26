import hashlib
from typing import Any

from app.contracts.common import ResponseMetadata, UsageMetadata
from app.contracts.llm import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse
from app.contracts.providers import ProviderMetadata
from app.core.request_context import get_request_id
from app.providers.base import LLMProvider


class MockLLMProvider(LLMProvider):
    name = "mock-llm"
    description = "Deterministic mock LLM provider"
    supports_chat = True
    supports_embeddings = True

    async def chat(self, request: ChatRequest) -> ChatResponse:
        last_user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        content = f"[mock-llm] deterministic response: {last_user_message}"
        return ChatResponse(
            content=content,
            metadata=ResponseMetadata(
                request_id=get_request_id(),
                provider=self.name,
                model=request.model or "mock-chat-v1",
            ),
            usage=UsageMetadata(input_tokens=len(request.messages), output_tokens=8, total_tokens=len(request.messages) + 8),
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        embeddings: list[list[float]] = [self._embedding_for_text(text) for text in request.texts]
        return EmbedResponse(
            embeddings=embeddings,
            metadata=ResponseMetadata(
                request_id=get_request_id(),
                provider=self.name,
                model=request.model or "mock-embed-v1",
            ),
        )

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "provider": self.name}

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            type="llm",
            description=self.description,
            capabilities=["chat", "embeddings"],
        )

    @staticmethod
    def _embedding_for_text(text: str, dim: int = 8) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [round(digest[i] / 255.0, 6) for i in range(dim)]
