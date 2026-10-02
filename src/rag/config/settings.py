"""Configuración centralizada de la aplicación.

Todos los parámetros se leen de variables de entorno (o de un archivo `.env`).
`get_settings()` devuelve siempre la misma instancia (patrón Singleton).
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Aplicación
    app_env: str = "development"
    log_level: str = "INFO"
    log_json: bool = True
    data_dir: Path = Path("data")

    # LLM (Ollama)
    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "llama3"
    llm_num_ctx: int = Field(8192, ge=512)
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(120.0, gt=0)
    llm_max_retries: int = Field(3, ge=0)

    # Embeddings y reranker
    embedding_model: str = "bge-m3"
    embedding_batch_size: int = Field(16, ge=1)
    embedding_timeout_seconds: float = Field(120.0, gt=0)
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"  # "cpu" o "cuda"
    reranker_max_length: int = Field(512, ge=64)

    # Recuperación
    hybrid_search_enabled: bool = True
    retrieval_top_k: int = Field(20, ge=1)
    # Cuántos candidatos (de los recuperados) puntúa el reranker; en CPU cada uno cuesta ~0,5 s
    rerank_candidates: int = Field(10, ge=1)
    rerank_top_n: int = Field(5, ge=1)
    rrf_k: int = Field(60, ge=1)
    min_relevance_score: float = Field(0.15, ge=0.0, le=1.0)

    # Chunking
    chunk_size: int = Field(800, ge=100)
    chunk_overlap: int = Field(100, ge=0)

    # ChromaDB
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "bancos"

    # PostgreSQL
    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "rag"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # Historial conversacional
    history_max_messages: int = Field(6, ge=0)
    max_question_length: int = Field(1000, ge=1)
    rate_limit_per_minute: int = Field(30, ge=1)

    # Scraping
    # NoDecode: la lista llega separada por comas (BANCO1,BANCO2), no como JSON
    scrape_banks: Annotated[list[str], NoDecode] = ["bbva", "bancolombia", "davivienda"]
    scrape_max_pages_per_bank: int = Field(250, ge=1)
    scrape_max_depth: int = Field(5, ge=0)
    scrape_delay_seconds: float = Field(1.0, ge=0)
    scrape_timeout_seconds: float = Field(20.0, gt=0)
    scrape_max_retries: int = Field(3, ge=0)
    scrape_user_agent: str = "rag-bancos-colombia/0.1 (proyecto academico)"

    @field_validator("scrape_banks", mode="before")
    @classmethod
    def _split_banks(cls, value: object) -> object:
        if isinstance(value, str):
            return [b.strip().lower() for b in value.split(",") if b.strip()]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE")
        if self.rerank_top_n > self.rerank_candidates:
            raise ValueError("RERANK_TOP_N no puede superar RERANK_CANDIDATES")
        if self.rerank_candidates > self.retrieval_top_k:
            raise ValueError("RERANK_CANDIDATES no puede superar RETRIEVAL_TOP_K")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración única de la aplicación."""
    return Settings()
