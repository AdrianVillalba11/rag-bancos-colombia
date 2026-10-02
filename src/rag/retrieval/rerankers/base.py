"""Estrategias de reordenamiento que no dependen de un modelo."""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.interfaces import Reranker
from rag.domain.models import RetrievedChunk


class NoopReranker(Reranker):
    """No reordena: conserva el orden recibido (reranker desactivado)."""

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return list(candidates[:top_n])
