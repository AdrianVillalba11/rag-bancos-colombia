"""Recuperación híbrida: búsqueda vectorial + BM25 fusionadas con RRF.

Reciprocal Rank Fusion combina rankings sin necesitar que sus puntajes sean comparables: cada
resultado suma 1 / (k + posición) por cada lista en la que aparece.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rag.domain.errors import RetrievalError
from rag.domain.interfaces import Embedder, VectorStore
from rag.domain.models import RetrievalSource, RetrievedChunk
from rag.retrieval.lexical import Bm25Index

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], k: int = 60, top_k: int | None = None
) -> list[RetrievedChunk]:
    puntajes: dict[str, float] = {}
    representantes: dict[str, RetrievedChunk] = {}
    similitudes: dict[str, float] = {}

    for ranking in rankings:
        for posicion, resultado in enumerate(ranking, start=1):
            clave = resultado.chunk.id
            puntajes[clave] = puntajes.get(clave, 0.0) + 1.0 / (k + posicion)
            representantes.setdefault(clave, resultado)
            if resultado.similarity is not None:
                similitudes[clave] = resultado.similarity

    ordenados = sorted(puntajes, key=lambda c: puntajes[c], reverse=True)
    fusionados = [
        RetrievedChunk(
            representantes[c].chunk, puntajes[c], RetrievalSource.HYBRID, similitudes.get(c)
        )
        for c in ordenados
    ]
    return fusionados[:top_k] if top_k else fusionados


class HybridRetriever:
    """Obtiene candidatos para una pregunta. Con `hybrid=False` usa solo la búsqueda vectorial."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        index: Bm25Index | None,
        *,
        top_k: int = 20,
        rrf_k: int = 60,
        hybrid: bool = True,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._index = index
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._hybrid = hybrid and index is not None

    def retrieve(self, query: str, bank: str | None = None) -> list[RetrievedChunk]:
        try:
            vector = self._store.search(self._embedder.embed_query(query), self._top_k, bank)
        except Exception as exc:
            raise RetrievalError(f"No se pudo recuperar contexto: {exc}", cause=exc) from exc
        if not self._hybrid:
            return vector
        assert self._index is not None
        lexico = self._index.search(query, self._top_k, bank)
        return reciprocal_rank_fusion([vector, lexico], self._rrf_k, self._top_k)
