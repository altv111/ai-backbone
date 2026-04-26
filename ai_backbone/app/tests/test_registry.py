import pytest

from app.contracts.providers import CollectionMetadata
from app.core.errors import ProviderNotFoundError
from app.core.registry import CollectionRegistry, LLMProviderRegistry
from app.providers.mock_llm import MockLLMProvider


def test_llm_registry_register_get_exists_list():
    registry = LLMProviderRegistry()
    provider = MockLLMProvider()
    registry.register(provider)

    assert registry.exists("mock-llm")
    assert registry.get("mock-llm") is provider
    assert len(list(registry.list())) == 1


def test_llm_registry_missing_provider_raises():
    registry = LLMProviderRegistry()
    with pytest.raises(ProviderNotFoundError):
        registry.get("missing")


def test_collection_registry_register_get_list():
    registry = CollectionRegistry()
    collection = CollectionMetadata(name="c1", provider="mock-rag")
    registry.register(collection)

    assert registry.exists("c1")
    assert registry.get("c1") == collection
    assert len(list(registry.list())) == 1
