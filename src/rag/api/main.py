"""Aplicación FastAPI: API de conversación y servidor de la interfaz web."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag import __version__
from rag.api.dependencies import AppState, build_app_state
from rag.api.errors import register_error_handlers
from rag.api.routes import analytics, chat, health
from rag.config import get_settings
from rag.infra.logging import configure_logging

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


def _warmup(state: AppState) -> None:
    """Carga los modelos en segundo plano para que la primera pregunta no pague el arranque."""
    try:
        state.service.warmup()
        logger.info("Modelos precargados")
    except Exception as exc:
        logger.warning("No se pudo precargar los modelos", extra={"error": str(exc)})
    finally:
        state.warm.set()


def create_app(state: AppState | None = None) -> FastAPI:
    """Crea la aplicación. En pruebas se inyecta un estado ya construido."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        estado = state
        if estado is None:
            settings = get_settings()
            configure_logging(settings.log_level, settings.log_json)
            estado = build_app_state(settings)
            threading.Thread(target=_warmup, args=(estado,), daemon=True).start()
        else:
            estado.warm.set()
        app.state.rag = estado
        yield
        estado.repository.close()

    app = FastAPI(
        title="Asistente RAG de bancos colombianos", version=__version__, lifespan=lifespan
    )
    register_error_handlers(app)
    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(analytics.router)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.middleware("http")
    async def limitar_tamano(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declarado = request.headers.get("content-length", "")
        maximo = request.app.state.rag.settings.max_request_bytes
        if declarado.isdigit() and int(declarado) > maximo:
            return JSONResponse(
                {"error": {"code": "payload_too_large", "message": "Solicitud demasiado grande"}},
                status_code=413,
            )
        return await call_next(request)

    @app.middleware("http")
    async def cabeceras_de_seguridad(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        respuesta = await call_next(request)
        respuesta.headers["Content-Security-Policy"] = _CSP
        respuesta.headers["X-Content-Type-Options"] = "nosniff"
        respuesta.headers["Referrer-Policy"] = "no-referrer"
        return respuesta

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "templates" / "index.html")

    @app.get("/analytics", include_in_schema=False)
    def analytics_page() -> FileResponse:
        return FileResponse(WEB_DIR / "templates" / "analytics.html")

    return app


app = create_app()
