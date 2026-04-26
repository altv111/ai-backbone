from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.health import router as health_router
from app.api.knowledge import router as knowledge_router
from app.api.llm import router as llm_router
from app.api.providers import router as providers_router
from app.api.rag import router as rag_router
from app.bootstrap import bootstrap_app
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.request_context import reset_request_id, set_request_id


app = FastAPI(title=settings.app_name, debug=settings.debug)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    token = set_request_id(request_id)
    try:
        response = await call_next(request)
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(health_router, prefix="/v1")
app.include_router(providers_router, prefix="/v1")
app.include_router(llm_router, prefix="/v1")
app.include_router(rag_router, prefix="/v1")
app.include_router(knowledge_router, prefix="/v1")


@app.on_event("startup")
async def on_startup() -> None:
    configure_logging(settings.debug)
    bootstrap_app(app, settings)


register_exception_handlers(app)
