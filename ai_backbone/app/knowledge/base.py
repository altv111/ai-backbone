from abc import ABC, abstractmethod
from typing import Any

from app.contracts.knowledge import (
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from app.contracts.providers import ProviderMetadata


class KnowledgeProvider(ABC):
    name: str
    description: str
    capabilities: list[str]

    @abstractmethod
    async def answer(self, request: KnowledgeAnswerRequest) -> KnowledgeAnswerResponse:
        raise NotImplementedError

    @abstractmethod
    async def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        raise NotImplementedError
