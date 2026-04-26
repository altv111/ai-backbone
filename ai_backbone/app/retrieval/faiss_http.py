import time
from typing import Any

import httpx

from app.contracts.common import ResponseMetadata
from app.contracts.providers import CollectionMetadata, ProviderMetadata
from app.contracts.rag import IndexRequest, IndexResponse, RetrievedChunk, RetrieveRequest, RetrieveResponse
from app.core.config import Settings
from app.core.errors import (
    InvalidProviderResponseError,
    ProviderFailedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.request_context import get_request_id
from app.retrieval.base import RetrievalProvider


class FaissHTTPRetrievalProvider(RetrievalProvider):
    name = "faiss-http"
    description = "HTTP retrieval provider backed by external FAISS worker service"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    async def index(self, request: IndexRequest) -> IndexResponse:
        worker_url = self._worker_url()
        payload = {
            "request_id": get_request_id(),
            "collection": request.collection,
            "documents": [
                {
                    "id": doc.id,
                    "text": doc.content,
                    "metadata": doc.metadata,
                }
                for doc in request.documents
            ],
            "mode": request.options.get("mode", "append"),
        }

        response_payload = await self._post_json(f"{worker_url}/index", payload)
        collection = response_payload.get("collection")
        indexed_count = response_payload.get("indexed_count")
        if not isinstance(collection, str) or not isinstance(indexed_count, int):
            raise InvalidProviderResponseError(
                "FAISS worker /index returned invalid response shape",
                {"provider": self.name},
            )

        return IndexResponse(
            collection=collection,
            indexed_count=indexed_count,
            metadata=ResponseMetadata(
                request_id=get_request_id(),
                provider=self.name,
                extra={
                    "worker_url": worker_url,
                    "total_count": response_payload.get("total_count"),
                    "worker_latency_ms": response_payload.get("latency_ms"),
                },
            ),
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        worker_url = self._worker_url()
        payload = {
            "request_id": get_request_id(),
            "collection": request.collection,
            "query": request.query,
            "top_k": request.top_k,
            "filters": request.filters,
        }

        response_payload = await self._post_json(f"{worker_url}/retrieve", payload)
        results = response_payload.get("results")
        if not isinstance(results, list):
            raise InvalidProviderResponseError(
                "FAISS worker /retrieve returned invalid response shape",
                {"provider": self.name},
            )

        chunks: list[RetrievedChunk] = []
        for result in results:
            if not isinstance(result, dict):
                raise InvalidProviderResponseError(
                    "FAISS worker /retrieve returned non-object result",
                    {"provider": self.name},
                )
            doc_id = result.get("id")
            text = result.get("text")
            score = result.get("score")
            metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
            if not isinstance(doc_id, str) or not isinstance(text, str) or not isinstance(score, (float, int)):
                raise InvalidProviderResponseError(
                    "FAISS worker /retrieve returned invalid result fields",
                    {"provider": self.name},
                )
            chunks.append(
                RetrievedChunk(
                    id=doc_id,
                    content=text,
                    score=float(score),
                    metadata=metadata,
                )
            )

        return RetrieveResponse(
            chunks=chunks,
            metadata=ResponseMetadata(
                request_id=get_request_id(),
                provider=self.name,
                extra={
                    "worker_url": worker_url,
                    "collection": request.collection,
                    "worker_latency_ms": response_payload.get("latency_ms"),
                },
            ),
        )

    async def health(self) -> dict[str, Any]:
        if not self._settings.faiss_worker_url:
            return {
                "status": "unavailable",
                "provider": self.name,
                "details": {"missing": ["FAISS_WORKER_URL"]},
            }

        try:
            payload = await self._get_json(f"{self._settings.faiss_worker_url.rstrip('/')}/health")
        except ProviderTimeoutError:
            return {
                "status": "degraded",
                "provider": self.name,
                "details": {"error": "timeout"},
            }
        except ProviderFailedError as exc:
            return {
                "status": "degraded",
                "provider": self.name,
                "details": {"error": str(exc)},
            }

        status = payload.get("status") if isinstance(payload, dict) else "unknown"
        return {
            "status": status if isinstance(status, str) else "unknown",
            "provider": self.name,
            "details": payload if isinstance(payload, dict) else {"raw": payload},
        }

    async def list_collections(self) -> list[CollectionMetadata]:
        if not self._settings.faiss_worker_url:
            return []

        payload = await self._get_json(f"{self._settings.faiss_worker_url.rstrip('/')}/collections")
        collections = payload.get("collections") if isinstance(payload, dict) else None
        if not isinstance(collections, list):
            raise InvalidProviderResponseError(
                "FAISS worker /collections returned invalid response shape",
                {"provider": self.name},
            )

        mapped: list[CollectionMetadata] = []
        for item in collections:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str):
                continue
            mapped.append(
                CollectionMetadata(
                    name=name,
                    provider=self.name,
                    description="FAISS worker collection",
                    metadata={
                        "document_count": item.get("document_count"),
                        "index_exists": item.get("index_exists"),
                        **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
                    },
                )
            )
        return mapped

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            type="retrieval",
            description=self.description,
            capabilities=["index", "retrieve", "collections"],
            metadata={
                "worker_url": self._settings.faiss_worker_url,
            },
        )

    def _worker_url(self) -> str:
        worker_url = self._settings.faiss_worker_url.strip()
        if not worker_url:
            raise ProviderUnavailableError(
                "Provider 'faiss-http' is not fully configured",
                {"missing": ["FAISS_WORKER_URL"], "provider": self.name},
            )
        return worker_url.rstrip("/")

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        request_id = get_request_id()
        if request_id:
            headers["X-Request-ID"] = request_id

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self._settings.faiss_http_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Provider '{self.name}' timed out",
                {"provider": self.name, "worker_url": url},
            ) from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderFailedError(
                f"Provider '{self.name}' returned status {response.status_code}",
                {
                    "provider": self.name,
                    "status_code": response.status_code,
                    "response": response.text[:500],
                    "worker_url": url,
                },
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise InvalidProviderResponseError(
                f"Provider '{self.name}' returned non-JSON response",
                {
                    "provider": self.name,
                    "worker_url": url,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                },
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidProviderResponseError(
                f"Provider '{self.name}' returned invalid JSON body",
                {"provider": self.name, "worker_url": url},
            )
        return payload

    async def _get_json(self, url: str) -> dict[str, Any]:
        headers = {}
        request_id = get_request_id()
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.faiss_http_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(url, headers=headers or None)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Provider '{self.name}' timed out",
                {"provider": self.name, "worker_url": url},
            ) from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderFailedError(
                f"Provider '{self.name}' returned status {response.status_code}",
                {
                    "provider": self.name,
                    "status_code": response.status_code,
                    "response": response.text[:500],
                    "worker_url": url,
                },
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise InvalidProviderResponseError(
                f"Provider '{self.name}' returned non-JSON response",
                {"provider": self.name, "worker_url": url},
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidProviderResponseError(
                f"Provider '{self.name}' returned invalid JSON body",
                {"provider": self.name, "worker_url": url},
            )
        return payload
