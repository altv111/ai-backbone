from fastapi import APIRouter, Request

from app.contracts.providers import CollectionsResponse, ProvidersResponse

router = APIRouter(tags=["providers"])


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(request: Request) -> ProvidersResponse:
    container = request.app.state.container
    return ProvidersResponse(
        llm=[provider.metadata() for provider in container.llm_registry.list()],
        retrieval=[provider.metadata() for provider in container.retrieval_registry.list()],
        knowledge=[provider.metadata() for provider in container.knowledge_registry.list()],
    )


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(request: Request) -> CollectionsResponse:
    container = request.app.state.container
    collections = list(container.collection_registry.list())
    known_names = {collection.name for collection in collections}

    for retrieval_provider in container.retrieval_registry.list():
        if hasattr(retrieval_provider, "list_collections"):
            try:
                dynamic_collections = await retrieval_provider.list_collections()
            except Exception:
                # Keep /collections resilient even when an external retrieval provider is unavailable.
                continue
            for collection in dynamic_collections:
                if collection.name not in known_names:
                    collections.append(collection)
                    known_names.add(collection.name)

    return CollectionsResponse(collections=collections)
