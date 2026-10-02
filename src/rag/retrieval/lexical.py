"""Índice léxico BM25, complemento de la búsqueda vectorial.

La búsqueda semántica entiende paráfrasis pero puede confundir nombres de producto, siglas y
cifras ("4x1000", "CDT", "Nequi"); BM25 las encuentra por coincidencia literal.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from rag.domain.models import Chunk, RetrievalSource, RetrievedChunk

_STOPWORDS = frozenset(
    "a al algo ante como con contra cual cuando de del desde donde el ella ellos en entre es esa "
    "ese eso esta estar este esto fue ha han hay la las le les lo los me mi mas muy no nos o "
    "para pero por que se si sin sobre son su sus te tu tus un una uno unos y ya yo".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(texto: str) -> list[str]:
    """Minúsculas, sin acentos, sin palabras vacías."""
    base = unicodedata.normalize("NFKD", texto.lower())
    base = "".join(c for c in base if not unicodedata.combining(c))
    return [t for t in _TOKEN.findall(base) if t not in _STOPWORDS and len(t) > 1]


class Bm25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus = [tokenize(c.embedding_text) for c in self._chunks]
        # BM25Okapi falla con un corpus vacío
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int, bank: str | None = None) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        puntajes = self._bm25.get_scores(tokens)
        ordenados = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)
        resultados: list[RetrievedChunk] = []
        for i in ordenados:
            if puntajes[i] <= 0 or len(resultados) >= top_k:
                break
            if bank and self._chunks[i].bank != bank:
                continue
            resultados.append(
                RetrievedChunk(self._chunks[i], float(puntajes[i]), RetrievalSource.LEXICAL)
            )
        return resultados
