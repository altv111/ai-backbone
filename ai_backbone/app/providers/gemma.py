import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.contracts.common import ResponseMetadata, UsageMetadata
from app.contracts.llm import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse
from app.contracts.providers import ProviderMetadata
from app.core.config import Settings
from app.core.errors import (
    InvalidProviderResponseError,
    ProviderBusyError,
    ProviderFailedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedOperationError,
)
from app.core.request_context import get_request_id
from app.providers._formatting import messages_to_text
from app.providers.base import LLMProvider


@dataclass
class GemmaWorker:
    id: str
    url: str
    max_concurrent: int
    active_requests: int = 0
    healthy: bool = True
    last_error: str | None = None
    avg_latency_ms: float | None = None


class GemmaWorkerPool:
    def __init__(self, workers: list[GemmaWorker]) -> None:
        self._workers = workers
        self._lock = asyncio.Lock()

    async def select_worker(self) -> GemmaWorker:
        async with self._lock:
            healthy_workers = [w for w in self._workers if w.healthy]
            if not healthy_workers:
                raise ProviderUnavailableError(
                    "No healthy local Gemma workers are available",
                    {"provider": "local-gemma"},
                )

            available = [w for w in healthy_workers if w.active_requests < w.max_concurrent]
            if not available:
                raise ProviderBusyError(
                    "All local Gemma workers are currently busy",
                    {"provider": "local-gemma"},
                )

            selected = sorted(
                available,
                key=lambda w: (
                    w.active_requests,
                    w.avg_latency_ms if w.avg_latency_ms is not None else float("inf"),
                ),
            )[0]
            selected.active_requests += 1
            return selected

    async def release_worker(
        self,
        worker: GemmaWorker,
        latency_ms: float | None = None,
        error: Exception | None = None,
    ) -> None:
        async with self._lock:
            worker.active_requests = max(worker.active_requests - 1, 0)
            if error is not None:
                worker.last_error = str(error)
                worker.healthy = False
            else:
                worker.last_error = None
                worker.healthy = True

            if latency_ms is not None:
                if worker.avg_latency_ms is None:
                    worker.avg_latency_ms = latency_ms
                else:
                    worker.avg_latency_ms = round((worker.avg_latency_ms * 0.7) + (latency_ms * 0.3), 3)

    async def health(self) -> dict[str, Any]:
        async with self._lock:
            worker_details = [
                {
                    "id": worker.id,
                    "url": worker.url,
                    "healthy": worker.healthy,
                    "active_requests": worker.active_requests,
                    "max_concurrent": worker.max_concurrent,
                    "avg_latency_ms": worker.avg_latency_ms,
                    "last_error": worker.last_error,
                }
                for worker in self._workers
            ]

        if not worker_details:
            status = "unavailable"
        elif all(item["healthy"] for item in worker_details):
            status = "ok"
        elif any(item["healthy"] for item in worker_details):
            status = "degraded"
        else:
            status = "unavailable"

        return {
            "status": status,
            "workers": worker_details,
        }


class LocalGemmaProvider(LLMProvider):
    name = "local-gemma"
    description = "Logical local Gemma provider backed by worker machines"
    supports_chat = True
    supports_embeddings = False
    supports_streaming = True

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport
        self._pool = GemmaWorkerPool(
            workers=[
                GemmaWorker(
                    id=worker["id"],
                    url=worker["url"],
                    # Keep CPU Gemma worker fan-in low; one active request per worker.
                    max_concurrent=1,
                )
                for worker in settings.gemma_workers_json
                if worker.get("id") and worker.get("url")
            ]
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if not self._pool._workers:
            raise ProviderUnavailableError(
                "No local Gemma workers are configured",
                {"provider": self.name},
            )

        queue_started = time.perf_counter()
        worker = await self._pool.select_worker()
        queue_wait_ms = round((time.perf_counter() - queue_started) * 1000.0, 3)
        req_id, payload = self._build_worker_payload(request)

        started = time.perf_counter()
        worker_error = None
        try:
            async with httpx.AsyncClient(
                timeout=self._settings.gemma_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{worker.url.rstrip('/')}/generate",
                    json=payload,
                )

            if response.status_code < 200 or response.status_code >= 300:
                raise ProviderFailedError(
                    f"Worker '{worker.id}' returned status {response.status_code}",
                    {
                        "provider": self.name,
                        "worker_id": worker.id,
                        "status_code": response.status_code,
                        "response": response.text.strip()[:500],
                    },
                )

            try:
                response_payload = response.json()
            except ValueError as exc:
                raise InvalidProviderResponseError(
                    f"Worker '{worker.id}' returned non-JSON response",
                    {"provider": self.name, "worker_id": worker.id},
                ) from exc
            content = response_payload.get("content") if isinstance(response_payload, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise InvalidProviderResponseError(
                    f"Worker '{worker.id}' returned invalid content",
                    {"provider": self.name, "worker_id": worker.id},
                )

            worker_latency_ms = response_payload.get("latency_ms") if isinstance(response_payload, dict) else None
            tokens_generated = response_payload.get("tokens_generated") if isinstance(response_payload, dict) else None
            model_name = response_payload.get("model_name") if isinstance(response_payload, dict) else None

            return ChatResponse(
                content=content,
                metadata=ResponseMetadata(
                    request_id=req_id,
                    provider=self.name,
                    model=model_name or payload["model_name"],
                    extra={
                        "worker_id": worker.id,
                        "queue_wait_ms": queue_wait_ms,
                        "worker_latency_ms": worker_latency_ms,
                        "tokens_generated": tokens_generated,
                    },
                ),
                usage=UsageMetadata(output_tokens=tokens_generated, total_tokens=tokens_generated),
            )
        except httpx.TimeoutException as exc:
            worker_error = exc
            raise ProviderTimeoutError(
                f"Worker '{worker.id}' timed out",
                {"provider": self.name, "worker_id": worker.id},
            ) from exc
        except (ProviderFailedError, InvalidProviderResponseError) as exc:
            worker_error = exc
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            await self._pool.release_worker(worker, latency_ms=elapsed_ms, error=worker_error)
            # TODO: In multi-process deployments, move worker utilization state to Redis or a real queue.

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        if not self._pool._workers:
            raise ProviderUnavailableError(
                "No local Gemma workers are configured",
                {"provider": self.name},
            )

        queue_started = time.perf_counter()
        worker = await self._pool.select_worker()
        queue_wait_ms = round((time.perf_counter() - queue_started) * 1000.0, 3)
        req_id, payload = self._build_worker_payload(request)
        started = time.perf_counter()
        worker_error = None

        async def _stream() -> AsyncIterator[str]:
            nonlocal worker_error
            try:
                headers = {}
                if req_id:
                    headers["X-Request-ID"] = req_id
                async with httpx.AsyncClient(
                    timeout=self._settings.gemma_timeout_seconds,
                    transport=self._transport,
                ) as client:
                    async with client.stream(
                        "POST",
                        f"{worker.url.rstrip('/')}/generate/stream",
                        json=payload,
                        headers=headers or None,
                    ) as response:
                        if response.status_code < 200 or response.status_code >= 300:
                            body = (await response.aread()).decode("utf-8", errors="ignore")
                            raise ProviderFailedError(
                                f"Worker '{worker.id}' returned status {response.status_code}",
                                {
                                    "provider": self.name,
                                    "worker_id": worker.id,
                                    "status_code": response.status_code,
                                    "response": body.strip()[:500],
                                },
                            )

                        yield _sse_event(
                            "start",
                            {
                                "provider": self.name,
                                "worker_id": worker.id,
                                "model_name": payload["model_name"],
                                "queue_wait_ms": queue_wait_ms,
                            },
                        )
                        async for chunk in response.aiter_text():
                            if chunk:
                                yield chunk
            except httpx.TimeoutException as exc:
                worker_error = exc
                raise ProviderTimeoutError(
                    f"Worker '{worker.id}' timed out",
                    {"provider": self.name, "worker_id": worker.id},
                ) from exc
            except (ProviderFailedError, InvalidProviderResponseError) as exc:
                worker_error = exc
                raise
            finally:
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
                await self._pool.release_worker(worker, latency_ms=elapsed_ms, error=worker_error)
                # TODO: In multi-process deployments, move worker utilization state to Redis or a real queue.

        return _stream()

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        raise UnsupportedOperationError(f"Provider '{self.name}' does not support embeddings")

    async def health(self) -> dict[str, Any]:
        health = await self._pool.health()
        return {
            "status": health["status"],
            "provider": self.name,
            "workers": health["workers"],
        }

    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self.name,
            type="llm",
            description=self.description,
            capabilities=["chat"],
            metadata={
                "default_model": self._settings.gemma_default_model,
                "workers": [worker["id"] for worker in self._settings.gemma_workers_json if worker.get("id")],
            },
        )

    def _build_worker_payload(self, request: ChatRequest) -> tuple[str | None, dict[str, Any]]:
        req_id = get_request_id()
        payload = {
            "request_id": req_id,
            "model_name": request.model or self._settings.gemma_default_model,
            "prompt": messages_to_text(request.messages),
            "temperature": request.temperature
            if request.temperature is not None
            else self._settings.gemma_default_temperature,
            "max_new_tokens": request.max_tokens
            if request.max_tokens is not None
            else self._settings.gemma_default_max_new_tokens,
        }
        return req_id, payload


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
