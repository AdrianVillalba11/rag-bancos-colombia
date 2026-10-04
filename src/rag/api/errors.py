"""Traducción de errores de dominio a respuestas HTTP coherentes."""

from __future__ import annotations

import logging
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag.domain.errors import RagError

logger = logging.getLogger(__name__)

_STATUS = {
    "invalid_input": 422,
    "unknown_bank": 422,
    "rate_limit_exceeded": 429,
    "service_busy": 503,
    "session_not_found": 404,
    "llm_unavailable": 503,
    "llm_response_error": 502,
    "vector_store_error": 503,
    "retrieval_error": 503,
    "embedding_error": 503,
    "conversation_repository_error": 503,
    "operation_timeout": 504,
}
# Errores cuyo mensaje es seguro mostrar tal cual (los redactamos nosotros)
_MENSAJE_PROPIO = {
    "invalid_input",
    "unknown_bank",
    "rate_limit_exceeded",
    "session_not_found",
    "service_busy",
}
GENERICO_503 = (
    "El servicio no está disponible en este momento. Inténtalo de nuevo en unos instantes."
)
GENERICO_500 = "Ocurrió un error inesperado. Inténtalo de nuevo."


def public_error(exc: Exception) -> tuple[int, str, str]:
    """(estado HTTP, código, mensaje seguro para el usuario). Nunca expone detalles internos."""
    if isinstance(exc, RagError):
        estado = _STATUS.get(exc.code, 500)
        if exc.code in _MENSAJE_PROPIO:
            return estado, exc.code, str(exc)
        return estado, exc.code, GENERICO_503 if estado >= 502 else GENERICO_500
    return 500, "internal_error", GENERICO_500


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagError)
    async def _rag_error(request: Request, exc: RagError) -> JSONResponse:
        estado, codigo, mensaje = public_error(exc)
        nivel = logging.WARNING if estado < 500 else logging.ERROR
        logger.log(nivel, "Error de dominio", extra={"code": codigo, "error": str(exc)})
        cabeceras = {}
        espera = getattr(exc, "retry_after", None)
        if espera is not None:
            cabeceras["Retry-After"] = str(max(1, math.ceil(espera)))
        return JSONResponse(
            {"error": {"code": codigo, "message": mensaje}}, status_code=estado, headers=cabeceras
        )

    @app.exception_handler(RequestValidationError)
    async def _validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "invalid_input", "message": "La solicitud no es válida"}},
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def _inesperado(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Error no controlado", extra={"path": request.url.path})
        return JSONResponse(
            {"error": {"code": "internal_error", "message": GENERICO_500}}, status_code=500
        )
