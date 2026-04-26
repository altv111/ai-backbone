from typing import Any, Optional

from pydantic import BaseModel, Field


class IndexDocument(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    request_id: Optional[str] = None
    collection: str
    documents: list[IndexDocument]
    mode: str = "append"


class IndexResponse(BaseModel):
    collection: str
    indexed_count: int
    total_count: int
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    request_id: Optional[str] = None
    collection: str
    query: str
    top_k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    collection: str
    results: list[RetrievedDocument]
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    name: str
    document_count: int
    index_exists: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    mock_mode: bool
    index_root: str
    collections_count: int
