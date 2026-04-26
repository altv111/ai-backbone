from typing import Optional

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    request_id: Optional[str] = None
    model_name: str
    prompt: str
    temperature: float = 0.2
    max_new_tokens: int = 512


class GenerateResponse(BaseModel):
    content: str
    model_name: str
    latency_ms: float
    tokens_generated: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_name: str
    loaded: bool
    active_requests: int
    max_concurrent: int
