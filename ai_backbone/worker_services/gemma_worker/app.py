import asyncio
import json
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from worker_services.gemma_worker.config import WorkerSettings, worker_settings
from worker_services.gemma_worker.model_loader import MockModelRunner, TransformersGemmaRunner
from worker_services.gemma_worker.schemas import GenerateRequest, GenerateResponse, HealthResponse


class WorkerState:
    def __init__(self, settings: WorkerSettings) -> None:
        self.settings = settings
        self.runner = None
        self.loaded = False
        self.active_requests = 0
        self.semaphore = asyncio.Semaphore(settings.max_concurrent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: WorkerState = app.state.worker_state
    if state.settings.mock_mode:
        state.runner = MockModelRunner()
    else:
        state.runner = TransformersGemmaRunner(
            model_name=state.settings.model_name,
            model_path=state.settings.model_path,
            num_threads=state.settings.num_threads,
            num_interop_threads=state.settings.num_interop_threads,
        )
    state.loaded = state.runner is not None
    yield


def create_app(settings: WorkerSettings | None = None) -> FastAPI:
    settings = settings or worker_settings
    app = FastAPI(title="gemma-worker", lifespan=lifespan)
    app.state.worker_state = WorkerState(settings=settings)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        state: WorkerState = app.state.worker_state
        return HealthResponse(
            status="ok" if state.loaded else "unavailable",
            model_name=state.settings.model_name,
            loaded=state.loaded,
            active_requests=state.active_requests,
            max_concurrent=state.settings.max_concurrent,
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        state: WorkerState = app.state.worker_state

        if state.semaphore.locked() and state.active_requests >= state.settings.max_concurrent:
            raise HTTPException(status_code=429, detail="Worker is busy")

        try:
            await asyncio.wait_for(state.semaphore.acquire(), timeout=0.01)
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail="Worker is busy") from exc

        state.active_requests += 1
        try:
            response = await state.runner.generate(payload)
            return response
        finally:
            state.active_requests = max(state.active_requests - 1, 0)
            state.semaphore.release()

    @app.post("/generate/stream")
    async def generate_stream(payload: GenerateRequest) -> StreamingResponse:
        state: WorkerState = app.state.worker_state

        if state.semaphore.locked() and state.active_requests >= state.settings.max_concurrent:
            raise HTTPException(status_code=429, detail="Worker is busy")

        try:
            await asyncio.wait_for(state.semaphore.acquire(), timeout=0.01)
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail="Worker is busy") from exc

        state.active_requests += 1

        async def _event_stream():
            started = time.perf_counter()
            approx_tokens = 0
            try:
                async for chunk in state.runner.generate_stream(payload):
                    if not chunk:
                        continue
                    approx_tokens += len(chunk.split())
                    yield _sse_event("delta", {"content": chunk})
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
                yield _sse_event(
                    "done",
                    {
                        "model_name": payload.model_name,
                        "latency_ms": elapsed_ms,
                        "tokens_generated": approx_tokens,
                    },
                )
            except Exception as exc:
                yield _sse_event("error", {"message": str(exc)})
            finally:
                state.active_requests = max(state.active_requests - 1, 0)
                state.semaphore.release()

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "worker_services.gemma_worker.app:app",
        host=worker_settings.host,
        port=worker_settings.port,
        reload=False,
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
