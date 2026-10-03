"""Composición de dependencias (inyección): construye los componentes concretos una sola vez.

Aquí, y solo aquí, se decide qué implementación concreta recibe cada contrato.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from rag.config import Settings, get_settings
from rag.conversation.memory import ConversationMemory
from rag.conversation.repository import PostgresConversationRepository
from rag.generation.llm import LLMFactory
from rag.generation.rag_service import RagService
from rag.infra.db import create_pool
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.indexer import build_embedder, build_vector_store
from rag.retrieval.lexical import Bm25Index
from rag.retrieval.rerankers import RerankerFactory

logger = logging.getLogger(__name__)


def build_rag_service(settings: Settings) -> RagService:
    store = build_vector_store(settings)
    index = None
    if settings.hybrid_search_enabled:
        chunks = store.list_chunks()
        index = Bm25Index(chunks)
        logger.info("Índice BM25 construido", extra={"chunks": index.size})
    retriever = HybridRetriever(
        build_embedder(settings),
        store,
        index,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.rrf_k,
        hybrid=settings.hybrid_search_enabled,
    )
    reranker = RerankerFactory.create(settings)
    return RagService(retriever, reranker, LLMFactory.create(settings), settings)


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


@lru_cache
def get_rag_service() -> RagService:
    return build_rag_service(get_settings())
