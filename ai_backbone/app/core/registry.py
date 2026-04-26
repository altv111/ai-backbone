from collections.abc import Iterable

from app.contracts.providers import CollectionMetadata
from app.core.errors import ProviderNotFoundError
from app.knowledge.base import KnowledgeProvider
from app.providers.base import LLMProvider
from app.retrieval.base import RetrievalProvider


class LLMProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            raise ProviderNotFoundError(f"Provider '{name}' was not found")
        return self._providers[name]

    def list(self) -> Iterable[LLMProvider]:
        return self._providers.values()

    def exists(self, name: str) -> bool:
        return name in self._providers


class RetrievalProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, RetrievalProvider] = {}

    def register(self, provider: RetrievalProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> RetrievalProvider:
        if name not in self._providers:
            raise ProviderNotFoundError(f"Provider '{name}' was not found")
        return self._providers[name]

    def list(self) -> Iterable[RetrievalProvider]:
        return self._providers.values()

    def exists(self, name: str) -> bool:
        return name in self._providers


class KnowledgeProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, KnowledgeProvider] = {}

    def register(self, provider: KnowledgeProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> KnowledgeProvider:
        if name not in self._providers:
            raise ProviderNotFoundError(f"Provider '{name}' was not found")
        return self._providers[name]

    def list(self) -> Iterable[KnowledgeProvider]:
        return self._providers.values()

    def exists(self, name: str) -> bool:
        return name in self._providers


class CollectionRegistry:
    def __init__(self) -> None:
        self._collections: dict[str, CollectionMetadata] = {}

    def register(self, collection: CollectionMetadata) -> None:
        self._collections[collection.name] = collection

    def get(self, name: str) -> CollectionMetadata:
        if name not in self._collections:
            raise ProviderNotFoundError(f"Collection '{name}' was not found")
        return self._collections[name]

    def list(self) -> Iterable[CollectionMetadata]:
        return self._collections.values()

    def exists(self, name: str) -> bool:
        return name in self._collections
