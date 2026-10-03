"""Estado de salud de la aplicación y de sus dependencias."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from rag.api.dependencies import AppState, get_state

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> JSONResponse:
    servicios = {
        "postgres": state.repository.ping(),
        "chroma": state.store.ping(),
        "ollama": state.llm.ping(),
    }
    indexados = 0
    if servicios["chroma"]:
        try:
            indexados = state.store.count()
        except Exception:
            servicios["chroma"] = False
    ok = all(servicios.values())
    cuerpo = {
        "status": "ok" if ok else "degradado",
        "services": servicios,
        "indexed_chunks": indexados,
        "warm": state.warm.is_set(),
    }
    return JSONResponse(cuerpo, status_code=200 if ok else 503)
