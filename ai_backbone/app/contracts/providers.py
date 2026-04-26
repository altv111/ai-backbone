from typing import Any

from pydantic import BaseModel, Field


class ProviderMetadata(BaseModel):
    name: str
    type: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProvidersResponse(BaseModel):
    llm: list[ProviderMetadata] = Field(default_factory=list)
    retrieval: list[ProviderMetadata] = Field(default_factory=list)
    knowledge: list[ProviderMetadata] = Field(default_factory=list)


class CollectionMetadata(BaseModel):
    name: str
    provider: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionsResponse(BaseModel):
    collections: list[CollectionMetadata] = Field(default_factory=list)
