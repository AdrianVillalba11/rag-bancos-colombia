"""Composición de dependencias (inyección): construye los componentes concretos una sola vez.

Aquí, y solo aquí, se decide qué implementación concreta recibe cada contrato.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from fastapi import Request

from rag.config import Settings
from rag.conversation.memory import ConversationMemory
from rag.conversation.repository import PostgresConversationRepository
from rag.domain.interfaces import LLMClient, VectorStore
from rag.generation.llm import LLMFactory
from rag.generation.rag_service import RagService
from rag.infra.db import create_pool
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.indexer import build_embedder, build_vector_store
from rag.retrieval.lexical import Bm25Index
from rag.retrieval.rerankers import RerankerFactory

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Servicios compartidos por todas las peticiones."""

    settings: Settings
    service: RagService
    memory: ConversationMemory
    repository: PostgresConversationRepository
    store: VectorStore
    llm: LLMClient
    warm: threading.Event = field(default_factory=threading.Event)


def _build_rag(settings: Settings) -> tuple[RagService, VectorStore, LLMClient]:
    store = build_vector_store(settings)
    index = None
    if settings.hybrid_search_enabled:
        index = Bm25Index(store.list_chunks())
        logger.info("Índice BM25 construido", extra={"chunks": index.size})
    retriever = HybridRetriever(
        build_embedder(settings),
        store,
        index,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.rrf_k,
        hybrid=settings.hybrid_search_enabled,
    )
    llm = LLMFactory.create(settings)
    service = RagService(retriever, RerankerFactory.create(settings), llm, settings)
    return service, store, llm


def build_rag_service(settings: Settings) -> RagService:
    return _build_rag(settings)[0]


def build_conversation_repository(settings: Settings) -> PostgresConversationRepository:
    """Crea el repositorio, abre el pool y garantiza el esquema."""
    repositorio = PostgresConversationRepository(
        create_pool(settings), max_retries=settings.llm_max_retries
    )
    repositorio.open()
    return repositorio


def build_memory(
    repositorio: PostgresConversationRepository, settings: Settings
) -> ConversationMemory:
    return ConversationMemory(repositorio, settings.history_max_messages)


def build_app_state(settings: Settings) -> AppState:
    repositorio = build_conversation_repository(settings)
    service, store, llm = _build_rag(settings)
    return AppState(
        settings=settings,
        service=service,
        memory=build_memory(repositorio, settings),
        repository=repositorio,
        store=store,
        llm=llm,
    )


def get_state(request: Request) -> AppState:
    """Dependencia de FastAPI: el estado creado al arrancar la aplicación."""
    return request.app.state.rag  # type: ignore[no-any-return]
