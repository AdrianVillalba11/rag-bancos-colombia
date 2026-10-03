"""Endpoints de conversación: chat (con streaming SSE), historial y feedback."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from rag.api.dependencies import AppState, get_state
from rag.api.errors import public_error
from rag.api.schemas import (
    BankOut,
    ChatRequest,
    ChatResponse,
    CitationOut,
    ConfigOut,
    FeedbackRequest,
    MessageOut,
    SessionOut,
    clean_question,
    resolve_bank,
)
from rag.conversation.memory import new_session_id, validate_session_id
from rag.domain.errors import InputValidationError
from rag.domain.models import BANK_NAMES, Citation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])
_MAX_SESIONES = 50


def _sse(evento: str, datos: dict[str, Any]) -> str:
    return f"event: {evento}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


def _citas(citations: tuple[Citation, ...]) -> list[dict[str, str]]:
    return [{"title": c.title, "url": c.url, "bank": c.bank} for c in citations]


@router.get("/config", response_model=ConfigOut)
def config(state: AppState = Depends(get_state)) -> ConfigOut:
    s = state.settings
    return ConfigOut(
        banks=[BankOut(id=b, name=BANK_NAMES.get(b, b.title())) for b in s.scrape_banks],
        max_question_length=s.max_question_length,
        history_max_messages=s.history_max_messages,
    )


@router.post("/chat", response_model=None)
def chat(req: ChatRequest, state: AppState = Depends(get_state)) -> Any:
    pregunta = clean_question(req.message, state.settings)
    banco = resolve_bank(req.bank, state.settings)
    sesion = validate_session_id(req.session_id) if req.session_id else new_session_id()

    # El historial se lee antes de guardar la pregunta actual para no duplicarla en el contexto
    historial = state.memory.history(sesion)
    state.memory.record_question(sesion, pregunta, banco)

    if not req.stream:
        final = state.service.answer(pregunta, historial, banco)
        mensaje_id = state.memory.record_answer(sesion, final.text, final.metrics, final.citations)
        return ChatResponse(
            session_id=sesion,
            message_id=mensaje_id,
            answer=final.text,
            answered=final.metrics.answered,
            citations=[CitationOut(**c) for c in _citas(final.citations)],
            latency_ms=final.metrics.latency_ms,
        )

    def eventos() -> Iterator[str]:
        yield _sse("meta", {"session_id": sesion})
        try:
            for evento in state.service.stream_answer(pregunta, historial, banco):
                if evento.token:
                    yield _sse("token", {"text": evento.token})
                if evento.final:
                    f = evento.final
                    mensaje_id = state.memory.record_answer(sesion, f.text, f.metrics, f.citations)
                    yield _sse(
                        "done",
                        {
                            "message_id": mensaje_id,
                            "answered": f.metrics.answered,
                            "citations": _citas(f.citations),
                            "latency_ms": f.metrics.latency_ms,
                        },
                    )
        except Exception as exc:  # el flujo ya empezó: el error viaja como evento
            _, codigo, mensaje = public_error(exc)
            logger.exception("Error durante el streaming", extra={"code": codigo})
            yield _sse("error", {"code": codigo, "message": mensaje})

    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions", response_model=list[SessionOut])
def sessions(ids: str = "", state: AppState = Depends(get_state)) -> list[SessionOut]:
    """Resumen de las conversaciones cuyos ID envía el cliente.

    No hay un listado global: el ID de sesión es la credencial de esa conversación, así que cada
    navegador solo puede consultar las suyas.
    """
    validos = []
    for candidato in ids.split(",")[:_MAX_SESIONES]:
        try:
            validos.append(validate_session_id(candidato.strip()))
        except InputValidationError:
            continue
    resumenes = state.repository.summarize_sessions(validos)
    return [
        SessionOut(
            session_id=s.session_id,
            title=s.title[:120],
            last_activity_at=s.last_activity_at.isoformat(),
            messages=s.messages,
        )
        for s in resumenes
    ]


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def session_messages(session_id: str, state: AppState = Depends(get_state)) -> list[MessageOut]:
    mensajes = state.repository.get_session(validate_session_id(session_id))
    return [
        MessageOut(
            id=m.id,
            role=m.role.value,
            content=m.content,
            created_at=m.created_at.isoformat(),
            citations=[CitationOut(title=c.title, url=c.url, bank=c.bank) for c in m.citations],
            feedback=m.feedback,
            answered=m.answered,
        )
        for m in mensajes
    ]


@router.post("/messages/{message_id}/feedback")
def feedback(
    message_id: int, req: FeedbackRequest, state: AppState = Depends(get_state)
) -> JSONResponse:
    state.repository.set_feedback(message_id, req.value)
    return JSONResponse({"ok": True})
