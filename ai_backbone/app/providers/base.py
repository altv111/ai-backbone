from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.contracts.llm import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse
from app.contracts.providers import ProviderMetadata


class LLMProvider(ABC):
    name: str
    description: str
    supports_chat: bool
    supports_embeddings: bool
    supports_streaming: bool = False

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    @abstractmethod
    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        raise NotImplementedError

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        raise NotImplementedError
