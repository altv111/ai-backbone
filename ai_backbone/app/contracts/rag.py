from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from app.contracts.common import ResponseMetadata


class DocumentInput(BaseModel):
    id: str
    content: str = Field(validation_alias=AliasChoices("content", "text"))
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    provider: str
    collection: str
    documents: list[DocumentInput]
    options: dict[str, Any] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    collection: str
    indexed_count: int
    metadata: ResponseMetadata


class RetrieveRequest(BaseModel):
    provider: str
    collection: str
    query: str
    top_k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    metadata: ResponseMetadata
