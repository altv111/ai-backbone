import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from worker_services.faiss_worker.config import FaissWorkerSettings, faiss_worker_settings
from worker_services.faiss_worker.embedding import MockEmbeddingService, SentenceTransformerEmbeddingService
from worker_services.faiss_worker.index_store import FaissIndexStore, MockIndexStore
from worker_services.faiss_worker.schemas import (
    CollectionsResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    RetrieveRequest,
    RetrieveResponse,
)


class FaissWorkerState:
    def __init__(self, settings: FaissWorkerSettings) -> None:
        self.settings = settings
        self.store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: FaissWorkerState = app.state.worker_state
    if state.settings.mock_mode:
        state.store = MockIndexStore()
    else:
        embedder = SentenceTransformerEmbeddingService(state.settings.embedding_model)
        state.store = FaissIndexStore(
            index_root=state.settings.index_root,
            embedding_model_name=state.settings.embedding_model,
            embedding_service=embedder,
        )
    yield


def create_app(settings: FaissWorkerSettings | None = None) -> FastAPI:
    settings = settings or faiss_worker_settings
    app = FastAPI(title="faiss-worker", lifespan=lifespan)
    app.state.worker_state = FaissWorkerState(settings=settings)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        state: FaissWorkerState = app.state.worker_state
        collections = await state.store.list_collections()
        return HealthResponse(
            status="ok",
            embedding_model=state.settings.embedding_model,
            mock_mode=state.settings.mock_mode,
            index_root=state.settings.index_root,
            collections_count=len(collections),
        )

    @app.post("/index", response_model=IndexResponse)
    async def index(payload: IndexRequest) -> IndexResponse:
        state: FaissWorkerState = app.state.worker_state

        if not payload.collection.strip():
            raise HTTPException(status_code=400, detail="collection is required")
        if payload.mode not in {"append", "replace"}:
            raise HTTPException(status_code=400, detail="mode must be append or replace")
        if not payload.documents:
            raise HTTPException(status_code=400, detail="documents must not be empty")
        if len(payload.documents) > state.settings.max_documents_per_index:
            raise HTTPException(
                status_code=400,
                detail=f"documents exceeds max limit ({state.settings.max_documents_per_index})",
            )

        for doc in payload.documents:
            if not doc.id.strip() or not doc.text.strip():
                raise HTTPException(status_code=400, detail="each document must have non-empty id and text")

        started = time.perf_counter()
        result = await state.store.index_documents(payload.collection, payload.documents, payload.mode)
        return IndexResponse(
            collection=result["collection"],
            indexed_count=result["indexed_count"],
            total_count=result["total_count"],
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            metadata=result.get("metadata", {}),
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
        state: FaissWorkerState = app.state.worker_state

        if not payload.collection.strip():
            raise HTTPException(status_code=400, detail="collection is required")
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="query is required")
        if payload.top_k < 1 or payload.top_k > state.settings.max_top_k:
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between 1 and {state.settings.max_top_k}",
            )

        started = time.perf_counter()
        try:
            results = await state.store.retrieve(payload.collection, payload.query, payload.top_k, payload.filters)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"collection '{payload.collection}' not found") from exc

        return RetrieveResponse(
            collection=payload.collection,
            results=results,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            metadata={"filters": payload.filters},
        )

    @app.get("/collections", response_model=CollectionsResponse)
    async def collections() -> CollectionsResponse:
        state: FaissWorkerState = app.state.worker_state
        return CollectionsResponse(collections=await state.store.list_collections())

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "worker_services.faiss_worker.app:app",
        host=faiss_worker_settings.host,
        port=faiss_worker_settings.port,
        reload=False,
    )
