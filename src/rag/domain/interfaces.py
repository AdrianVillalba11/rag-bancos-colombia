"""Contratos del dominio.

Las capas superiores dependen de estas abstracciones y no de implementaciones concretas, lo que
permite intercambiar scrapers, rerankers o modelos (patrón Strategy) y aislar el acceso a
almacenamiento (patrón Repository).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from rag.domain.models import (
    AnswerMetrics,
    Chunk,
    Citation,
    Message,
    RawPage,
    RetrievedChunk,
    StoredMessage,
)


class Scraper(ABC):
    """Estrategia de extracción para el sitio de un banco."""

    #: Identificador corto del banco, por ejemplo "bbva"
    bank: str

    @abstractmethod
    def scrape(self) -> Iterator[RawPage]:
        """Recorre el sitio y produce las páginas descargadas."""


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(ABC):
    """Repositorio de chunks indexados."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Inserta o actualiza chunks. Debe ser idempotente por `Chunk.id`."""

    @abstractmethod
    def search(
        self, query_embedding: Sequence[float], top_k: int, bank: str | None = None
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def list_chunks(self, bank: str | None = None) -> list[Chunk]:
        """Devuelve todos los chunks (necesario para el índice léxico BM25)."""

    @abstractmethod
    def delete_bank(self, bank: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def ping(self) -> bool: ...


class Reranker(ABC):
    """Estrategia de reordenamiento de candidatos según su relevancia para la pregunta."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class LLMClient(ABC):
    """Estrategia de generación de texto."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...

    @abstractmethod
    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Produce la respuesta por fragmentos a medida que se genera."""

    @abstractmethod
    def ping(self) -> bool: ...


class ConversationRepository(ABC):
    """Repositorio del historial de conversaciones."""

    @abstractmethod
    def add_user_message(self, session_id: str, content: str, bank_filter: str | None) -> int: ...

    @abstractmethod
    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metrics: AnswerMetrics,
        citations: Sequence[Citation],
    ) -> int: ...

    @abstractmethod
    def get_recent(self, session_id: str, limit: int) -> list[Message]:
        """Últimos `limit` mensajes de la sesión, en orden cronológico."""

    @abstractmethod
    def get_session(self, session_id: str) -> list[StoredMessage]: ...

    @abstractmethod
    def list_sessions(self, limit: int = 50) -> list[str]: ...

    @abstractmethod
    def set_feedback(self, message_id: int, value: int) -> None: ...

    @abstractmethod
    def iter_messages(self) -> Iterator[StoredMessage]:
        """Recorre todo el histórico, para el módulo de analítica."""

    @abstractmethod
    def ping(self) -> bool: ...
