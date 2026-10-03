"""Entidades del dominio. Son datos puros: no dependen de ninguna librería de infraestructura."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

#: Nombre legible de cada banco soportado
BANK_NAMES = {
    "bbva": "BBVA Colombia",
    "bancolombia": "Bancolombia",
    "davivienda": "Davivienda",
}


def utcnow() -> datetime:
    return datetime.now(UTC)


def content_hash(text: str) -> str:
    """Hash estable del contenido, usado para deduplicar páginas y generar IDs idempotentes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Ingesta ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class RawPage:
    """Página tal como se descargó del sitio, sin procesar."""

    bank: str
    url: str
    html: str
    status_code: int = 200
    fetched_at: datetime = field(default_factory=utcnow)

    @property
    def hash(self) -> str:
        return content_hash(self.html)


@dataclass(frozen=True)
class Section:
    """Fragmento estructural de un documento (encabezado y su texto)."""

    heading: str
    text: str


@dataclass(frozen=True)
class CleanDocument:
    """Contenido útil de una página, ya depurado y organizado por secciones."""

    bank: str
    url: str
    title: str
    sections: tuple[Section, ...]
    fetched_at: datetime
    source_hash: str

    @property
    def id(self) -> str:
        return content_hash(f"{self.bank}|{self.url}")[:16]

    @property
    def text(self) -> str:
        return "\n\n".join(
            f"{s.heading}\n{s.text}".strip() if s.heading else s.text for s in self.sections
        )


@dataclass(frozen=True)
class Chunk:
    """Unidad que se vectoriza e indexa."""

    document_id: str
    bank: str
    url: str
    title: str
    heading: str
    text: str
    position: int

    @property
    def id(self) -> str:
        """ID determinista: reindexar el mismo contenido no crea duplicados."""
        return content_hash(f"{self.document_id}|{self.position}|{self.text}")[:24]

    @property
    def embedding_text(self) -> str:
        """Texto que se vectoriza e indexa: el contenido precedido por su contexto."""
        contexto = " — ".join(p for p in (self.title, self.heading) if p)
        return f"{contexto}\n{self.text}" if contexto else self.text

    def metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "bank": self.bank,
            "url": self.url,
            "title": self.title,
            "heading": self.heading,
            "position": self.position,
        }


# --- Recuperación -----------------------------------------------------------------------------
class RetrievalSource(StrEnum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    RERANKED = "reranked"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    source: RetrievalSource = RetrievalSource.VECTOR
    #: Similitud coseno con la pregunta, si el chunk salió de la búsqueda vectorial (0 a 1)
    similarity: float | None = None


@dataclass(frozen=True)
class Citation:
    """Referencia a la fuente que respalda una respuesta."""

    title: str
    url: str
    bank: str


# --- Conversación -----------------------------------------------------------------------------
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True)
class SessionSummary:
    """Resumen de una conversación para listarla en el historial."""

    session_id: str
    title: str  # primera pregunta del usuario
    last_activity_at: datetime
    messages: int


@dataclass(frozen=True)
class StoredMessage:
    """Mensaje persistido, con el contexto necesario para analítica."""

    id: int
    session_id: str
    role: Role
    content: str
    created_at: datetime
    bank_filter: str | None = None
    latency_ms: int | None = None
    retrieval_ms: int | None = None
    rerank_ms: int | None = None
    generation_ms: int | None = None
    top_score: float | None = None
    answered: bool | None = None
    rewritten_query: str | None = None
    citations: tuple[Citation, ...] = ()
    feedback: int | None = None  # 1 = útil, -1 = no útil


@dataclass(frozen=True)
class AnswerMetrics:
    """Mediciones de una respuesta, que alimentan la analítica."""

    latency_ms: int
    retrieval_ms: int = 0
    rerank_ms: int = 0
    generation_ms: int = 0
    top_score: float | None = None
    answered: bool = True
    rewritten_query: str | None = None


@dataclass(frozen=True)
class RagAnswer:
    """Resultado completo de responder una pregunta."""

    text: str
    citations: tuple[Citation, ...]
    metrics: AnswerMetrics
    chunks: tuple[RetrievedChunk, ...] = ()


@dataclass(frozen=True)
class StreamEvent:
    """Evento de una respuesta en streaming: un fragmento de texto o el resultado final."""

    token: str | None = None
    final: RagAnswer | None = None
