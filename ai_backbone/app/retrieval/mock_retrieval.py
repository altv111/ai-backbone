from collections import defaultdict
from typing import Any

from app.contracts.common import ResponseMetadata
from app.contracts.providers import ProviderMetadata
from app.contracts.rag import (
    DocumentInput,
    IndexRequest,
    IndexResponse,
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
)
from app.core.request_context import get_request_id
from app.retrieval.base import RetrievalProvider


class MockRetrievalProvider(RetrievalProvider):
    name = "mock-rag"
    description = "In-memory mock retrieval provider"

    def __init__(self) -> None:
        self._docs: dict[str, list[DocumentInput]] = defaultdict(list)

    async def index(self, request: IndexRequest) -> IndexResponse:
        self._docs[request.collection].extend(request.documents)
        return IndexResponse(
            collection=request.collection,
            indexed_count=len(request.documents),
            metadata=ResponseMetadata(
                request_id=get_request_id(),
                provider=self.name,
            ),
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        docs = self._docs.get(request.collection, [])
        ranked = self._rank_documents(docs, request.query)
        chunks = [
            RetrievedChunk(
                id=doc.id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
            )
            for doc, score in ranked[: request.top_k]
        ]
        return RetrieveResponse(
            chunks=chunks,
            metadata=ResponseMetadata(
                request_id=get_request_id(),
                provider=self.name,
            ),
        )

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "provider": self.name}

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            type="retrieval",
            description=self.description,
            capabilities=["index", "retrieve"],
        )

    @staticmethod
    def _rank_documents(documents: list[DocumentInput], query: str) -> list[tuple[DocumentInput, float]]:
        query_terms = {term for term in query.lower().split() if term}
        scored: list[tuple[DocumentInput, float]] = []
        for idx, doc in enumerate(documents):
            content_terms = set(doc.content.lower().split())
            overlap = len(query_terms & content_terms)
            if query_terms:
                score = overlap / len(query_terms)
            else:
                score = 0.0
            scored.append((doc, score if score > 0 else max(0.0, 0.1 - idx * 0.001)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
