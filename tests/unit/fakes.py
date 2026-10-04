"""Dobles de prueba compartidos por los tests de la API."""

from datetime import datetime

from rag.domain.errors import SessionNotFoundError
from rag.domain.models import (
    AnswerMetrics,
    Citation,
    Message,
    RagAnswer,
    Role,
    SessionSummary,
    StoredMessage,
    StreamEvent,
)


class RepoFalso:
    """Repositorio en memoria con el mismo contrato que el de Postgres."""

    def __init__(self):
        self.filas: list[dict] = []
        self.feedback: dict[int, int] = {}
        self.vivo = True

    def _agregar(self, session_id, role, content, **extra):
        fila = {"id": len(self.filas) + 1, "session_id": session_id, "role": role,
                "content": content, **extra}  # fmt: skip
        self.filas.append(fila)
        return fila["id"]

    def add_user_message(self, session_id, content, bank_filter):
        return self._agregar(session_id, Role.USER, content, bank=bank_filter)

    def add_assistant_message(self, session_id, content, metrics, citations):
        return self._agregar(session_id, Role.ASSISTANT, content, citations=tuple(citations),
                             answered=metrics.answered)  # fmt: skip

    def get_recent(self, session_id, limit):
        propios = [f for f in self.filas if f["session_id"] == session_id]
        return [Message(f["role"], f["content"]) for f in propios][-limit:] if limit else []

    def get_session(self, session_id):
        return [
            StoredMessage(
                id=f["id"],
                session_id=session_id,
                role=f["role"],
                content=f["content"],
                created_at=datetime(2026, 1, 1),
                citations=f.get("citations", ()),
                feedback=self.feedback.get(f["id"]),
                answered=f.get("answered"),
            )
            for f in self.filas
            if f["session_id"] == session_id
        ]

    def summarize_sessions(self, session_ids):
        resumenes = []
        for sid in session_ids:
            propios = [f for f in self.filas if f["session_id"] == sid]
            if not propios:
                continue
            titulo = next(f["content"] for f in propios if f["role"] == Role.USER)
            fecha = datetime(2026, 1, 1)
            resumenes.append(SessionSummary(sid, titulo, fecha, len(propios)))
        return resumenes

    def iter_messages(self):
        for sid in dict.fromkeys(f["session_id"] for f in self.filas):
            yield from self.get_session(sid)

    def set_feedback(self, message_id, value):
        if message_id not in {f["id"] for f in self.filas if f["role"] == Role.ASSISTANT}:
            raise SessionNotFoundError(f"No existe la respuesta {message_id}")
        self.feedback[message_id] = value

    def ping(self):
        return self.vivo

    def close(self):
        pass


class ServicioFalso:
    def __init__(self):
        self.llamadas: list[tuple] = []
        self.falla: Exception | None = None

    def _final(self, texto="Necesitas cédula [1]."):
        citas = (Citation("Cuenta de ahorros", "https://x.com/a", "davivienda"),)
        return RagAnswer(texto, citas, AnswerMetrics(latency_ms=123, answered=True))

    def stream_answer(self, question, history, bank):
        self.llamadas.append((question, list(history), bank))
        yield StreamEvent(token="Necesitas ")
        if self.falla:
            raise self.falla
        yield StreamEvent(token="cédula [1].")
        yield StreamEvent(final=self._final())

    def answer(self, question, history, bank):
        self.llamadas.append((question, list(history), bank))
        return self._final()


class StoreFalso:
    def ping(self):
        return True

    def count(self):
        return 3235


class LLMFalso:
    def ping(self):
        return True
