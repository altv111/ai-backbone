from typing import Any, Optional

from pydantic import BaseModel, Field

from app.contracts.common import ResponseMetadata, SourceReference


class KnowledgeAnswerRequest(BaseModel):
    provider: str
    query: str
    instructions: Optional[str] = None
    options: dict[str, Any] = Field(default_factory=dict)


class KnowledgeAnswerResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    metadata: ResponseMetadata


class KnowledgeSearchRequest(BaseModel):
    provider: str
    query: str
    top_k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchResponse(BaseModel):
    sources: list[SourceReference]
    metadata: ResponseMetadata
