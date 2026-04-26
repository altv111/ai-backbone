from typing import Any, Optional

from pydantic import BaseModel, Field


class ResponseMetadata(BaseModel):
    request_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class UsageMetadata(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class SourceReference(BaseModel):
    id: str
    title: Optional[str] = None
    content: Optional[str] = None
    score: Optional[float] = None
    source_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
