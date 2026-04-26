from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_name: str = Field(
        default="gemma-12b-it",
        validation_alias=AliasChoices("GEMMA_WORKER_MODEL_NAME", "AI_BACKBONE_GEMMA_WORKER_MODEL_NAME"),
    )
    model_path: str = Field(
        default="",
        validation_alias=AliasChoices("GEMMA_WORKER_MODEL_PATH", "AI_BACKBONE_GEMMA_WORKER_MODEL_PATH"),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("GEMMA_WORKER_HOST", "AI_BACKBONE_GEMMA_WORKER_HOST"),
    )
    port: int = Field(
        default=8888,
        validation_alias=AliasChoices("GEMMA_WORKER_PORT", "AI_BACKBONE_GEMMA_WORKER_PORT"),
    )
    max_concurrent: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("GEMMA_WORKER_MAX_CONCURRENT", "AI_BACKBONE_GEMMA_WORKER_MAX_CONCURRENT"),
    )
    num_threads: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("GEMMA_WORKER_NUM_THREADS", "AI_BACKBONE_GEMMA_WORKER_NUM_THREADS"),
    )
    num_interop_threads: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "GEMMA_WORKER_NUM_INTEROP_THREADS",
            "AI_BACKBONE_GEMMA_WORKER_NUM_INTEROP_THREADS",
        ),
    )
    mock_mode: bool = Field(
        default=True,
        validation_alias=AliasChoices("GEMMA_WORKER_MOCK_MODE", "AI_BACKBONE_GEMMA_WORKER_MOCK_MODE"),
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("max_concurrent", mode="after")
    @classmethod
    def enforce_single_concurrency(cls, value: int) -> int:
        # CPU-only Gemma workers should run one active generation at a time.
        return 1


worker_settings = WorkerSettings()
