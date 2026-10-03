"""Memoria conversacional: ventana de los últimos N mensajes de una sesión."""

from __future__ import annotations

import re
import uuid

from rag.domain.errors import InputValidationError
from rag.domain.interfaces import ConversationRepository
from rag.domain.models import AnswerMetrics, Citation, Message

_SESION_VALIDA = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def new_session_id() -> str:
    return uuid.uuid4().hex


def validate_session_id(session_id: str) -> str:
    """Los ID de sesión los elige el cliente: se restringe su formato antes de usarlos."""
    if not _SESION_VALIDA.match(session_id or ""):
        raise InputValidationError(
            "El ID de sesión debe tener entre 8 y 64 caracteres (letras, números, - y _)"
        )
    return session_id


class ConversationMemory:
    """Une el repositorio con la política de memoria (cuántos mensajes previos se recuerdan)."""

    def __init__(self, repository: ConversationRepository, max_messages: int) -> None:
        self._repository = repository
        self._max_messages = max_messages

    @property
    def max_messages(self) -> int:
        return self._max_messages

    def history(self, session_id: str) -> list[Message]:
        """Últimos N mensajes de la sesión (vacío si N es 0 o la sesión es nueva)."""
        return self._repository.get_recent(validate_session_id(session_id), self._max_messages)

    def record_question(self, session_id: str, question: str, bank: str | None) -> int:
        return self._repository.add_user_message(validate_session_id(session_id), question, bank)

    def record_answer(
        self,
        session_id: str,
        text: str,
        metrics: AnswerMetrics,
        citations: tuple[Citation, ...],
    ) -> int:
        return self._repository.add_assistant_message(
            validate_session_id(session_id), text, metrics, citations
        )
