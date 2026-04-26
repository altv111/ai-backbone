from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FaissWorkerSettings(BaseSettings):
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("FAISS_WORKER_HOST", "AI_BACKBONE_FAISS_WORKER_HOST"),
    )
    port: int = Field(
        default=8890,
        validation_alias=AliasChoices("FAISS_WORKER_PORT", "AI_BACKBONE_FAISS_WORKER_PORT"),
    )
    index_root: str = Field(
        default="./data/faiss_indexes",
        validation_alias=AliasChoices("FAISS_WORKER_INDEX_ROOT", "AI_BACKBONE_FAISS_WORKER_INDEX_ROOT"),
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("FAISS_WORKER_EMBEDDING_MODEL", "AI_BACKBONE_FAISS_WORKER_EMBEDDING_MODEL"),
    )
    mock_mode: bool = Field(
        default=True,
        validation_alias=AliasChoices("FAISS_WORKER_MOCK_MODE", "AI_BACKBONE_FAISS_WORKER_MOCK_MODE"),
    )
    max_documents_per_index: int = Field(
        default=10000,
        ge=1,
        validation_alias=AliasChoices(
            "FAISS_WORKER_MAX_DOCUMENTS_PER_INDEX",
            "AI_BACKBONE_FAISS_WORKER_MAX_DOCUMENTS_PER_INDEX",
        ),
    )
    max_top_k: int = Field(
        default=50,
        ge=1,
        validation_alias=AliasChoices("FAISS_WORKER_MAX_TOP_K", "AI_BACKBONE_FAISS_WORKER_MAX_TOP_K"),
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


faiss_worker_settings = FaissWorkerSettings()
