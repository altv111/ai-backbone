from fastapi import APIRouter, Request

from app.contracts.rag import IndexRequest, IndexResponse, RetrieveRequest, RetrieveResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/index", response_model=IndexResponse)
async def index(request: Request, payload: IndexRequest) -> IndexResponse:
    return await request.app.state.container.rag_service.index(payload)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: Request, payload: RetrieveRequest) -> RetrieveResponse:
    return await request.app.state.container.rag_service.retrieve(payload)
