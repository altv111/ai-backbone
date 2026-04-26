from typing import Any, Optional

from pydantic import BaseModel, Field

from app.contracts.common import ResponseMetadata, UsageMetadata


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    options: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    content: str
    metadata: ResponseMetadata
    usage: Optional[UsageMetadata] = None


class EmbedRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    texts: list[str]
    options: dict[str, Any] = Field(default_factory=dict)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    metadata: ResponseMetadata
