import time

from app.contracts.knowledge import (
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from app.core.config import Settings
from app.core.errors import AppError, InvalidRequestError, ProviderFailedError
from app.core.registry import KnowledgeProviderRegistry


class KnowledgeService:
    def __init__(self, registry: KnowledgeProviderRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings

    async def answer(self, request: KnowledgeAnswerRequest) -> KnowledgeAnswerResponse:
        provider = self.registry.get(request.provider)
        start = time.perf_counter()
        try:
            response = await provider.answer(request)
        except AppError:
            raise
        except Exception as exc:
            raise ProviderFailedError(f"Provider '{provider.name}' failed while answering") from exc
        response.metadata.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return response

    async def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        if request.top_k > self.settings.max_top_k:
            raise InvalidRequestError(
                "top_k exceeds allowed maximum",
                {"max_top_k": self.settings.max_top_k},
            )

        provider = self.registry.get(request.provider)
        start = time.perf_counter()
        try:
            response = await provider.search(request)
        except AppError:
            raise
        except Exception as exc:
            raise ProviderFailedError(f"Provider '{provider.name}' failed while searching") from exc
        response.metadata.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return response
