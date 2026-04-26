from fastapi import APIRouter, Request

from app.contracts.knowledge import (
    KnowledgeAnswerRequest,
    KnowledgeAnswerResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/answer", response_model=KnowledgeAnswerResponse)
async def answer(request: Request, payload: KnowledgeAnswerRequest) -> KnowledgeAnswerResponse:
    return await request.app.state.container.knowledge_service.answer(payload)


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search(request: Request, payload: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    return await request.app.state.container.knowledge_service.search(payload)
