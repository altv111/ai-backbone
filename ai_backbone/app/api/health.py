from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    app_name = request.app.state.container.settings.app_name
    return {"status": "ok", "app": app_name}


@router.get("/health/providers")
async def providers_health(request: Request) -> dict:
    container = request.app.state.container

    async def _safe_provider_health(provider):
        try:
            details = await provider.health()
            status = details.get("status", "ok") if isinstance(details, dict) else "ok"
            return {
                "name": provider.name,
                "status": status,
                "details": details if isinstance(details, dict) else {"raw": details},
            }
        except Exception as exc:
            return {
                "name": provider.name,
                "status": "unavailable",
                "details": {"error": str(exc)},
            }

    llm = [await _safe_provider_health(provider) for provider in container.llm_registry.list()]
    retrieval = [await _safe_provider_health(provider) for provider in container.retrieval_registry.list()]
    knowledge = [await _safe_provider_health(provider) for provider in container.knowledge_registry.list()]

    return {
        "status": "ok",
        "providers": {
            "llm": llm,
            "retrieval": retrieval,
            "knowledge": knowledge,
        },
    }
