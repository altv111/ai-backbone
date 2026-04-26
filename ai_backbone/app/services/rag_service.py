import time

from app.contracts.providers import CollectionMetadata
from app.contracts.rag import IndexRequest, IndexResponse, RetrieveRequest, RetrieveResponse
from app.core.config import Settings
from app.core.errors import AppError, InvalidRequestError, ProviderFailedError
from app.core.registry import CollectionRegistry, RetrievalProviderRegistry


class RAGService:
    def __init__(
        self,
        retrieval_registry: RetrievalProviderRegistry,
        collection_registry: CollectionRegistry,
        settings: Settings,
    ) -> None:
        self.retrieval_registry = retrieval_registry
        self.collection_registry = collection_registry
        self.settings = settings

    async def index(self, request: IndexRequest) -> IndexResponse:
        if len(request.documents) > self.settings.max_documents_per_index:
            raise InvalidRequestError(
                "Too many documents in index request",
                {"max_documents_per_index": self.settings.max_documents_per_index},
            )

        provider = self.retrieval_registry.get(request.provider)

        if not self.collection_registry.exists(request.collection):
            self.collection_registry.register(
                CollectionMetadata(
                    name=request.collection,
                    provider=request.provider,
                    description="Auto-registered collection",
                )
            )

        start = time.perf_counter()
        try:
            response = await provider.index(request)
        except AppError:
            raise
        except Exception as exc:
            raise ProviderFailedError(f"Provider '{provider.name}' failed while indexing documents") from exc
        response.metadata.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return response

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        if request.top_k > self.settings.max_top_k:
            raise InvalidRequestError(
                "top_k exceeds allowed maximum",
                {"max_top_k": self.settings.max_top_k},
            )

        provider = self.retrieval_registry.get(request.provider)

        start = time.perf_counter()
        try:
            response = await provider.retrieve(request)
        except AppError:
            raise
        except Exception as exc:
            raise ProviderFailedError(f"Provider '{provider.name}' failed while retrieving documents") from exc
        response.metadata.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return response
