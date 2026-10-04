"""Indexación de chunks: vectoriza y los guarda en la base vectorial."""

from __future__ import annotations

import logging

from rag.config import Settings
from rag.domain.interfaces import Embedder, VectorStore
from rag.domain.models import Chunk
from rag.retrieval.embeddings import OllamaEmbedder
from rag.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class ChunkIndexer:
    """Reemplaza el contenido indexado de un banco por sus chunks actuales (idempotente)."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def __call__(self, bank: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectores = self._embedder.embed_documents([c.embedding_text for c in chunks])
        self._store.delete_bank(bank)
        self._store.upsert(chunks, vectores)
        logger.info("Banco indexado", extra={"bank": bank, "chunks": len(chunks)})


def build_embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(
        settings.ollama_base_url,
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def build_vector_store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(
        settings.chroma_host,
        settings.chroma_port,
        settings.chroma_collection,
        max_retries=settings.llm_max_retries,
        timeout=settings.chroma_timeout_seconds,
    )


def build_indexer(settings: Settings) -> ChunkIndexer:
    return ChunkIndexer(build_embedder(settings), build_vector_store(settings))
