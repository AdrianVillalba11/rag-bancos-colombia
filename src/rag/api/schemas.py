"""Esquemas de entrada y salida de la API (validación con pydantic)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from rag.config import Settings
from rag.domain.errors import InputValidationError, UnknownBankError

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TODOS = {"", "todos", "todo", "all"}


class ChatRequest(BaseModel):
    message: str = Field(..., description="Pregunta del usuario")
    session_id: str | None = Field(None, description="ID de la conversación; se crea si falta")
    bank: str | None = Field(None, description="Limita la consulta a un banco")
    stream: bool = Field(True, description="Responder por streaming (SSE)")


class FeedbackRequest(BaseModel):
    value: Literal[1, -1]


class CitationOut(BaseModel):
    title: str
    url: str
    bank: str


class ChatResponse(BaseModel):
    session_id: str
    message_id: int
    answer: str
    answered: bool
    citations: list[CitationOut]
    latency_ms: int


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str
    citations: list[CitationOut] = []
    feedback: int | None = None
    answered: bool | None = None


class SessionOut(BaseModel):
    session_id: str
    title: str
    last_activity_at: str
    messages: int


class BankOut(BaseModel):
    id: str
    name: str


class ConfigOut(BaseModel):
    banks: list[BankOut]
    max_question_length: int
    history_max_messages: int


def clean_question(texto: str, settings: Settings) -> str:
    """Valida la pregunta: sin caracteres de control, no vacía y dentro del límite."""
    limpio = _CONTROL.sub("", texto or "").strip()
    if not limpio:
        raise InputValidationError("La pregunta no puede estar vacía")
    if len(limpio) > settings.max_question_length:
        raise InputValidationError(
            f"La pregunta supera el máximo de {settings.max_question_length} caracteres"
        )
    return limpio


def resolve_bank(bank: str | None, settings: Settings) -> str | None:
    if bank is None or bank.strip().lower() in _TODOS:
        return None
    valor = bank.strip().lower()
    if valor not in settings.scrape_banks:
        raise UnknownBankError(f"Banco no disponible: {bank!r}")
    return valor
