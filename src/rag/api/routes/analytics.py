"""Endpoint de analítica del histórico de conversaciones."""

from __future__ import annotations

import hmac
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import JSONResponse

from rag.analytics.metrics import compute_report
from rag.api.dependencies import AppState, get_state

router = APIRouter(prefix="/api", tags=["analytics"])


def _autorizado(state: AppState, token: str | None) -> bool:
    esperado = state.settings.analytics_token
    if not esperado:
        return True  # sin token configurado el endpoint es abierto (uso local)
    return token is not None and hmac.compare_digest(token, esperado)


@router.get("/analytics", response_model=None)
def analytics(
    state: Annotated[AppState, Depends(get_state)],
    days: Annotated[int | None, Query(ge=1, le=3650, description="Últimos N días")] = None,
    x_analytics_token: Annotated[str | None, Header()] = None,
) -> Any:
    if not _autorizado(state, x_analytics_token):
        return JSONResponse(
            {"error": {"code": "unauthorized", "message": "Token de analítica inválido"}},
            status_code=401,
        )
    desde = datetime.now(UTC) - timedelta(days=days) if days else None
    return compute_report(
        state.repository.iter_messages(),
        since=desde,
        manual_minutes=state.settings.analytics_manual_search_minutes,
    )


@router.get("/analytics/protected")
def analytics_protected(state: Annotated[AppState, Depends(get_state)]) -> dict[str, bool]:
    """Indica a la interfaz si el dashboard exige token."""
    return {"protected": bool(state.settings.analytics_token)}
