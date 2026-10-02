"""Embeddings con bge-m3 servido por Ollama."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

import httpx

from rag.domain.errors import EmbeddingError
from rag.domain.interfaces import Embedder
from rag.infra.retry import retry

logger = logging.getLogger(__name__)


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        batch_size: int = 16,
        timeout: float = 120.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._embed_batch = retry(
            max_retries=max_retries,
            base_delay=1.0,
            retry_on=(EmbeddingError,),
            should_retry=lambda exc: getattr(exc, "retryable", False),
            sleep=sleep,
        )(self._embed_batch_once)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectores: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            vectores.extend(self._embed_batch(list(texts[i : i + self._batch_size])))
        return vectores

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch_once(self, textos: list[str]) -> list[list[float]]:
        try:
            respuesta = self._client.post(
                "/api/embed", json={"model": self._model, "input": textos}
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"No se pudo contactar con Ollama: {exc}", cause=exc) from exc
        if respuesta.status_code != 200:
            error = EmbeddingError(
                f"Ollama respondió HTTP {respuesta.status_code}: {respuesta.text[:200]}"
            )
            error.retryable = respuesta.status_code >= 500 or respuesta.status_code == 429
            raise error
        vectores = respuesta.json().get("embeddings")
        if not vectores or len(vectores) != len(textos):
            error = EmbeddingError("Ollama devolvió una cantidad de embeddings inesperada")
            error.retryable = False
            raise error
        return vectores
