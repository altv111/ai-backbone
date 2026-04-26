from abc import ABC, abstractmethod
from typing import Any

from app.contracts.providers import ProviderMetadata
from app.contracts.rag import IndexRequest, IndexResponse, RetrieveRequest, RetrieveResponse


class RetrievalProvider(ABC):
    name: str
    description: str

    @abstractmethod
    async def index(self, request: IndexRequest) -> IndexResponse:
        raise NotImplementedError

    @abstractmethod
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        raise NotImplementedError
