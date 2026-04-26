from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.contracts.llm import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    return await request.app.state.container.llm_service.chat(payload)


@router.post("/embed", response_model=EmbedResponse)
async def embed(request: Request, payload: EmbedRequest) -> EmbedResponse:
    return await request.app.state.container.llm_service.embed(payload)


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    stream = await request.app.state.container.llm_service.chat_stream(payload)
    return StreamingResponse(stream, media_type="text/event-stream")
