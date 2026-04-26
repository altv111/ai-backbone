from typing import Any

from app.contracts.common import ResponseMetadata, SourceReference
from app.contracts.knowledge import (
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from app.contracts.providers import ProviderMetadata
from app.core.request_context import get_request_id
from app.knowledge.base import KnowledgeProvider


class MockKnowledgeProvider(KnowledgeProvider):
    name = "mock-knowledge"
    description = "Deterministic mock knowledge provider"
    capabilities = ["answer", "search"]

    async def answer(self, request: KnowledgeAnswerRequest) -> KnowledgeAnswerResponse:
        sources = [
            SourceReference(
                id="mock-source-1",
                title="Mock Knowledge Source",
                content=f"Reference content for query: {request.query}",
                score=0.95,
                source_type="mock",
                metadata={"provider": self.name},
            )
        ]
        return KnowledgeAnswerResponse(
            answer=f"[mock-knowledge] deterministic answer for: {request.query}",
            sources=sources,
            metadata=ResponseMetadata(request_id=get_request_id(), provider=self.name),
        )

    async def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        sources = [
            SourceReference(
                id=f"mock-search-{i + 1}",
                title=f"Mock Result {i + 1}",
                content=f"Result for query '{request.query}'",
                score=1.0 - i * 0.05,
                source_type="mock",
                metadata={"provider": self.name},
            )
            for i in range(request.top_k)
        ]
        return KnowledgeSearchResponse(
            sources=sources,
            metadata=ResponseMetadata(request_id=get_request_id(), provider=self.name),
        )

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "provider": self.name}

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            type="knowledge",
            description=self.description,
            capabilities=self.capabilities,
        )
