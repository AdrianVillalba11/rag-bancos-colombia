"""Caché de embeddings precalculados por banco.

Vectorizar los ~3.200 chunks con `bge-m3` tarda unos 3 minutos con GPU y más de 25 en CPU. Como el
contenido limpio está versionado, también se versionan sus embeddings (`data/clean/<banco>/
embeddings.npz`, ~6 MB en total, en media precisión): el primer arranque indexa en segundos y sin
usar el modelo de embeddings.

La caché solo se usa si coinciden el modelo y el texto exacto de cada chunk (se compara una huella
del texto que se vectoriza); en cualquier otro caso se ignora y se recalcula.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rag.domain.models import Chunk, content_hash

logger = logging.getLogger(__name__)


def chunk_fingerprints(chunks: Sequence[Chunk]) -> list[str]:
    """Huella del texto que realmente se vectoriza (título + encabezado + contenido)."""
    return [content_hash(c.embedding_text)[:20] for c in chunks]


class EmbeddingCache:
    def __init__(self, clean_dir: Path, model: str) -> None:
        self._clean_dir = clean_dir
        self._model = model

    def _path(self, bank: str) -> Path:
        return self._clean_dir / bank / "embeddings.npz"

    def load(self, bank: str, chunks: Sequence[Chunk]) -> list[list[float]] | None:
        """Devuelve los vectores guardados si siguen siendo válidos para estos chunks."""
        ruta = self._path(bank)
        if not ruta.exists():
            return None
        try:
            with np.load(ruta, allow_pickle=False) as datos:
                modelo = str(datos["model"])
                huellas = [str(x) for x in datos["fingerprints"]]
                vectores = datos["vectors"]
        except Exception as exc:  # archivo corrupto o de otro formato
            logger.warning("Caché de embeddings ilegible", extra={"bank": bank, "error": str(exc)})
            return None

        if modelo != self._model:
            logger.info("Caché de embeddings de otro modelo", extra={"bank": bank, "model": modelo})
            return None
        if huellas != chunk_fingerprints(chunks) or len(vectores) != len(chunks):
            logger.info("Caché de embeddings desactualizada", extra={"bank": bank})
            return None
        return vectores.astype(np.float32).tolist()  # type: ignore[no-any-return]

    def save(self, bank: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> bool:
        """Guarda los vectores. Si el directorio es de solo lectura, no es un error."""
        ruta = self._path(bank)
        try:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ruta,
                model=np.array(self._model),
                fingerprints=np.array(chunk_fingerprints(chunks)),
                vectors=np.asarray(vectors, dtype=np.float16),
            )
        except OSError as exc:
            logger.info(
                "No se pudo guardar la caché de embeddings", extra={"bank": bank, "error": str(exc)}
            )
            return False
        return True
