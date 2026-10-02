"""Reranker cross-encoder bge-reranker-v2-m3.

A diferencia de la búsqueda vectorial (que compara embeddings calculados por separado), un
cross-encoder lee la pregunta y el fragmento juntos y estima su relevancia con mucha más
precisión; por eso solo se aplica a los pocos candidatos que ya preseleccionó la recuperación.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from rag.domain.errors import RerankerError
from rag.domain.interfaces import Reranker
from rag.domain.models import RetrievalSource, RetrievedChunk

logger = logging.getLogger(__name__)


class BgeReranker(Reranker):
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        max_length: int = 512,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._loader = model_loader
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                if self._loader is not None:
                    self._model = self._loader()
                else:
                    from sentence_transformers import CrossEncoder  # carga perezosa: es pesado

                    logger.info("Cargando reranker", extra={"model": self._model_name})
                    opciones: dict[str, Any] = {}
                    if self._device.startswith("cuda"):
                        import torch

                        opciones["torch_dtype"] = torch.float16  # mitad de VRAM
                    self._model = CrossEncoder(
                        self._model_name,
                        device=self._device,
                        max_length=self._max_length,
                        model_kwargs=opciones,
                    )
            except Exception as exc:
                raise RerankerError(
                    f"No se pudo cargar el reranker {self._model_name!r}: {exc}", cause=exc
                ) from exc
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        modelo = self._load()
        pares = [(query, c.chunk.embedding_text) for c in candidates]
        try:
            # CrossEncoder.predict ya devuelve probabilidades de relevancia entre 0 y 1
            puntajes = modelo.predict(pares)
        except Exception as exc:
            raise RerankerError(f"Fallo al puntuar candidatos: {exc}", cause=exc) from exc

        reordenados = [
            RetrievedChunk(
                c.chunk, min(1.0, max(0.0, float(s))), RetrievalSource.RERANKED, c.similarity
            )
            for c, s in zip(candidates, puntajes, strict=True)
        ]
        reordenados.sort(key=lambda r: r.score, reverse=True)
        return reordenados[:top_n]
