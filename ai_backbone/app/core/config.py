from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ai-backbone"
    environment: str = "dev"
    debug: bool = False
    max_messages: int = Field(default=50, ge=1)
    max_texts_per_embed: int = Field(default=100, ge=1)
    max_documents_per_index: int = Field(default=100, ge=1)
    max_top_k: int = Field(default=50, ge=1)
    enable_mock_providers: bool = True
    audit_enabled: bool = True
    audit_log_path: str = ""

    central_llm_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("CENTRAL_LLM_ENABLED", "AI_BACKBONE_CENTRAL_LLM_ENABLED"),
    )
    central_llm_api_url: str = Field(
        default="",
        validation_alias=AliasChoices("CENTRAL_LLM_API_URL", "AI_BACKBONE_CENTRAL_LLM_API_URL"),
    )
    central_llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CENTRAL_LLM_API_KEY", "AI_BACKBONE_CENTRAL_LLM_API_KEY"),
    )
    central_llm_timeout_seconds: float = Field(
        default=60,
        ge=1,
        validation_alias=AliasChoices("CENTRAL_LLM_TIMEOUT_SECONDS", "AI_BACKBONE_CENTRAL_LLM_TIMEOUT_SECONDS"),
    )
    central_llm_default_model: str = Field(
        default="gemini-flash",
        validation_alias=AliasChoices("CENTRAL_LLM_DEFAULT_MODEL", "AI_BACKBONE_CENTRAL_LLM_DEFAULT_MODEL"),
    )
    central_llm_allowed_models: list[str] = Field(
        default_factory=lambda: ["gemini-flash", "gemini-pro"],
        validation_alias=AliasChoices("CENTRAL_LLM_ALLOWED_MODELS", "AI_BACKBONE_CENTRAL_LLM_ALLOWED_MODELS"),
    )
    central_llm_default_data_classification: str = Field(
        default="internal",
        validation_alias=AliasChoices(
            "CENTRAL_LLM_DEFAULT_DATA_CLASSIFICATION",
            "AI_BACKBONE_CENTRAL_LLM_DEFAULT_DATA_CLASSIFICATION",
        ),
    )
    central_llm_verify_ssl: bool = Field(
        default=False,
        validation_alias=AliasChoices("CENTRAL_LLM_VERIFY_SSL", "AI_BACKBONE_CENTRAL_LLM_VERIFY_SSL"),
    )

    gemma_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("GEMMA_ENABLED", "AI_BACKBONE_GEMMA_ENABLED"),
    )
    gemma_workers_json: list[dict] = Field(
        default_factory=list,
        validation_alias=AliasChoices("GEMMA_WORKERS_JSON", "AI_BACKBONE_GEMMA_WORKERS_JSON"),
    )
    gemma_timeout_seconds: float = Field(
        default=120,
        ge=1,
        validation_alias=AliasChoices("GEMMA_TIMEOUT_SECONDS", "AI_BACKBONE_GEMMA_TIMEOUT_SECONDS"),
    )
    gemma_default_model: str = Field(
        default="gemma-12b-it",
        validation_alias=AliasChoices("GEMMA_DEFAULT_MODEL", "AI_BACKBONE_GEMMA_DEFAULT_MODEL"),
    )
    gemma_default_max_new_tokens: int = Field(
        default=512,
        ge=1,
        validation_alias=AliasChoices("GEMMA_DEFAULT_MAX_NEW_TOKENS", "AI_BACKBONE_GEMMA_DEFAULT_MAX_NEW_TOKENS"),
    )
    gemma_default_temperature: float = Field(
        default=0.2,
        validation_alias=AliasChoices("GEMMA_DEFAULT_TEMPERATURE", "AI_BACKBONE_GEMMA_DEFAULT_TEMPERATURE"),
    )

    faiss_http_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("FAISS_HTTP_ENABLED", "AI_BACKBONE_FAISS_HTTP_ENABLED"),
    )
    faiss_worker_url: str = Field(
        default="http://localhost:8890",
        validation_alias=AliasChoices("FAISS_WORKER_URL", "AI_BACKBONE_FAISS_WORKER_URL"),
    )
    faiss_http_timeout_seconds: float = Field(
        default=120,
        ge=1,
        validation_alias=AliasChoices("FAISS_HTTP_TIMEOUT_SECONDS", "AI_BACKBONE_FAISS_HTTP_TIMEOUT_SECONDS"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AI_BACKBONE_",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("central_llm_allowed_models", mode="before")
    @classmethod
    def _parse_allowed_models(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


settings = Settings()
