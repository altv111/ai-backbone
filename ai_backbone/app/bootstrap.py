from dataclasses import dataclass

from fastapi import FastAPI

from app.contracts.providers import CollectionMetadata
from app.core.audit import AuditService
from app.core.config import Settings
from app.core.registry import (
    CollectionRegistry,
    KnowledgeProviderRegistry,
    LLMProviderRegistry,
    RetrievalProviderRegistry,
)
from app.knowledge.mock_knowledge import MockKnowledgeProvider
from app.providers.central_llm import CentralLLMProvider
from app.providers.gemma import LocalGemmaProvider
from app.providers.mock_llm import MockLLMProvider
from app.retrieval.faiss_http import FaissHTTPRetrievalProvider
from app.retrieval.mock_retrieval import MockRetrievalProvider
from app.services.knowledge_service import KnowledgeService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


@dataclass
class AppContainer:
    settings: Settings
    llm_registry: LLMProviderRegistry
    retrieval_registry: RetrievalProviderRegistry
    knowledge_registry: KnowledgeProviderRegistry
    collection_registry: CollectionRegistry
    audit_service: AuditService
    llm_service: LLMService
    rag_service: RAGService
    knowledge_service: KnowledgeService


def _safe_register_llm(registry: LLMProviderRegistry, provider) -> None:
    if registry.exists(provider.name):
        raise ValueError(f"Duplicate LLM provider name: {provider.name}")
    registry.register(provider)


def build_container(settings: Settings) -> AppContainer:
    llm_registry = LLMProviderRegistry()
    retrieval_registry = RetrievalProviderRegistry()
    knowledge_registry = KnowledgeProviderRegistry()
    collection_registry = CollectionRegistry()
    audit_service = AuditService(enabled=settings.audit_enabled, log_path=settings.audit_log_path)

    if settings.enable_mock_providers:
        _safe_register_llm(llm_registry, MockLLMProvider())
        retrieval_registry.register(MockRetrievalProvider())
        knowledge_registry.register(MockKnowledgeProvider())

        collection_registry.register(
            CollectionMetadata(
                name="default",
                provider="mock-rag",
                description="Default in-memory collection",
            )
        )

    if settings.central_llm_enabled:
        _safe_register_llm(llm_registry, CentralLLMProvider(settings=settings))

    if settings.gemma_enabled:
        _safe_register_llm(llm_registry, LocalGemmaProvider(settings=settings))

    if settings.faiss_http_enabled:
        retrieval_registry.register(FaissHTTPRetrievalProvider(settings=settings))

    return AppContainer(
        settings=settings,
        llm_registry=llm_registry,
        retrieval_registry=retrieval_registry,
        knowledge_registry=knowledge_registry,
        collection_registry=collection_registry,
        audit_service=audit_service,
        llm_service=LLMService(llm_registry, settings, audit_service),
        rag_service=RAGService(retrieval_registry, collection_registry, settings),
        knowledge_service=KnowledgeService(knowledge_registry, settings),
    )


def bootstrap_app(app: FastAPI, settings: Settings) -> None:
    app.state.container = build_container(settings)
