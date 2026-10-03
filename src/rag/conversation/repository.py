"""Repositorio de conversaciones sobre PostgreSQL (patrón Repository)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from rag.conversation.schema import SCHEMA_LOCK_ID, SCHEMA_STATEMENTS
from rag.domain.errors import ConversationRepositoryError, SessionNotFoundError
from rag.domain.interfaces import ConversationRepository
from rag.domain.models import (
    AnswerMetrics,
    Citation,
    Message,
    Role,
    SessionSummary,
    StoredMessage,
)
from rag.infra.retry import retry

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SELECCION_MENSAJES = """
    SELECT m.*,
           COALESCE(
               json_agg(json_build_object('title', c.title, 'url', c.url, 'bank', c.bank)
                        ORDER BY c.position) FILTER (WHERE c.message_id IS NOT NULL),
               '[]'::json
           ) AS citations
    FROM messages m
    LEFT JOIN message_citations c ON c.message_id = m.id
"""


def _stored(fila: dict[str, Any]) -> StoredMessage:
    return StoredMessage(
        id=fila["id"],
        session_id=fila["session_id"],
        role=Role(fila["role"]),
        content=fila["content"],
        created_at=fila["created_at"],
        bank_filter=fila["bank_filter"],
        latency_ms=fila["latency_ms"],
        retrieval_ms=fila["retrieval_ms"],
        rerank_ms=fila["rerank_ms"],
        generation_ms=fila["generation_ms"],
        top_score=fila["top_score"],
        answered=fila["answered"],
        rewritten_query=fila["rewritten_query"],
        citations=tuple(Citation(c["title"], c["url"], c["bank"]) for c in fila["citations"]),
        feedback=fila["feedback"],
    )


class PostgresConversationRepository(ConversationRepository):
    def __init__(
        self, pool: ConnectionPool, *, max_retries: int = 3, open_timeout: float = 15.0
    ) -> None:
        self._pool = pool
        self._open_timeout = open_timeout
        self._reintentar = retry(
            max_retries=max_retries,
            base_delay=0.5,
            retry_on=(_Transitorio,),
        )

    # --- Infraestructura ----------------------------------------------------------------------
    def open(self) -> None:
        """Abre el pool y crea el esquema si hace falta."""
        self._ejecutar("No se pudo abrir el pool de conexiones", self._abrir)

    def close(self) -> None:
        self._pool.close()

    def _abrir(self) -> None:
        self._pool.open(wait=True, timeout=self._open_timeout)
        with self._pool.connection() as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (SCHEMA_LOCK_ID,))
            try:
                for sentencia in SCHEMA_STATEMENTS:
                    conn.execute(sentencia)
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (SCHEMA_LOCK_ID,))

    def _ejecutar(self, descripcion: str, operacion: Callable[[], T]) -> T:
        """Ejecuta una operación traduciendo los errores de la base a errores de dominio."""

        def intento() -> T:
            try:
                return operacion()
            except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
                raise _Transitorio(f"{descripcion}: {exc}") from exc
            except psycopg.Error as exc:
                raise ConversationRepositoryError(f"{descripcion}: {exc}", cause=exc) from exc

        try:
            return self._reintentar(intento)()
        except _Transitorio as exc:
            raise ConversationRepositoryError(str(exc), cause=exc) from exc

    # --- Escritura ----------------------------------------------------------------------------
    def add_user_message(self, session_id: str, content: str, bank_filter: str | None) -> int:
        def op() -> int:
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO conversations (session_id) VALUES (%s) "
                    "ON CONFLICT (session_id) DO UPDATE SET last_activity_at = now()",
                    (session_id,),
                )
                fila = conn.execute(
                    "INSERT INTO messages (session_id, role, content, bank_filter) "
                    "VALUES (%s, 'user', %s, %s) RETURNING id",
                    (session_id, content, bank_filter),
                ).fetchone()
                return int(fila[0])  # type: ignore[index]

        return self._ejecutar("No se pudo guardar el mensaje del usuario", op)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metrics: AnswerMetrics,
        citations: Sequence[Citation],
    ) -> int:
        def op() -> int:
            with self._pool.connection() as conn:
                fila = conn.execute(
                    "INSERT INTO messages (session_id, role, content, latency_ms, retrieval_ms, "
                    "rerank_ms, generation_ms, top_score, answered, rewritten_query) "
                    "VALUES (%s, 'assistant', %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (
                        session_id,
                        content,
                        metrics.latency_ms,
                        metrics.retrieval_ms,
                        metrics.rerank_ms,
                        metrics.generation_ms,
                        metrics.top_score,
                        metrics.answered,
                        metrics.rewritten_query,
                    ),
                ).fetchone()
                mensaje_id = int(fila[0])  # type: ignore[index]
                for posicion, c in enumerate(citations):
                    conn.execute(
                        "INSERT INTO message_citations (message_id, position, title, url, bank) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (mensaje_id, posicion, c.title, c.url, c.bank),
                    )
                conn.execute(
                    "UPDATE conversations SET last_activity_at = now() WHERE session_id = %s",
                    (session_id,),
                )
                return mensaje_id

        return self._ejecutar("No se pudo guardar la respuesta", op)

    def set_feedback(self, message_id: int, value: int) -> None:
        if value not in (-1, 1):
            raise ConversationRepositoryError("El feedback debe ser 1 o -1")

        def op() -> None:
            with self._pool.connection() as conn:
                n = conn.execute(
                    "UPDATE messages SET feedback = %s WHERE id = %s AND role = 'assistant'",
                    (value, message_id),
                ).rowcount
                if n == 0:
                    raise SessionNotFoundError(f"No existe la respuesta {message_id}")

        self._ejecutar("No se pudo guardar el feedback", op)

    # --- Lectura ------------------------------------------------------------------------------
    def get_recent(self, session_id: str, limit: int) -> list[Message]:
        if limit <= 0:
            return []

        def op() -> list[Message]:
            with self._pool.connection() as conn:
                filas = conn.execute(
                    "SELECT role, content, created_at FROM messages "
                    "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                    (session_id, limit),
                ).fetchall()
            return [Message(Role(r), c, t) for r, c, t in reversed(filas)]

        return self._ejecutar("No se pudo leer el historial", op)

    def get_session(self, session_id: str) -> list[StoredMessage]:
        def op() -> list[StoredMessage]:
            with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    _SELECCION_MENSAJES + " WHERE m.session_id = %s GROUP BY m.id ORDER BY m.id",
                    (session_id,),
                )
                return [_stored(f) for f in cur.fetchall()]

        return self._ejecutar("No se pudo leer la conversación", op)

    def list_sessions(self, limit: int = 50) -> list[str]:
        def op() -> list[str]:
            with self._pool.connection() as conn:
                filas = conn.execute(
                    "SELECT session_id FROM conversations ORDER BY last_activity_at DESC LIMIT %s",
                    (limit,),
                ).fetchall()
            return [f[0] for f in filas]

        return self._ejecutar("No se pudo listar las sesiones", op)

    def summarize_sessions(self, session_ids: Sequence[str]) -> list[SessionSummary]:
        if not session_ids:
            return []

        def op() -> list[SessionSummary]:
            with self._pool.connection() as conn:
                filas = conn.execute(
                    "SELECT c.session_id, c.last_activity_at, "
                    "  COALESCE((SELECT m.content FROM messages m WHERE m.session_id = "
                    "    c.session_id AND m.role = 'user' ORDER BY m.id LIMIT 1), '') AS title, "
                    "  (SELECT count(*) FROM messages m WHERE m.session_id = c.session_id) AS n "
                    "FROM conversations c WHERE c.session_id = ANY(%s) "
                    "ORDER BY c.last_activity_at DESC",
                    (list(session_ids),),
                ).fetchall()
            return [SessionSummary(s, t, a, int(n)) for s, a, t, n in filas]

        return self._ejecutar("No se pudo resumir las sesiones", op)

    def iter_messages(self) -> Iterator[StoredMessage]:
        """Recorre todo el histórico en orden cronológico, sin cargarlo completo en memoria."""
        try:
            with self._pool.connection() as conn, conn.cursor(
                name="historico", row_factory=dict_row
            ) as cur:
                cur.itersize = 500
                cur.execute(_SELECCION_MENSAJES + " GROUP BY m.id ORDER BY m.id")
                for fila in cur:
                    yield _stored(fila)
        except psycopg.Error as exc:
            raise ConversationRepositoryError(
                f"No se pudo recorrer el histórico: {exc}", cause=exc
            ) from exc

    def ping(self) -> bool:
        try:
            with self._pool.connection(timeout=3) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False


class _Transitorio(Exception):
    """Fallo de conexión con la base de datos que puede reintentarse."""
