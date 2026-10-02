"""Repositorio de chunks sobre ChromaDB (patrón Repository)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from rag.domain.errors import VectorStoreError
from rag.domain.interfaces import VectorStore
from rag.domain.models import Chunk, RetrievalSource, RetrievedChunk
from rag.infra.retry import retry

logger = logging.getLogger(__name__)

T = TypeVar("T")
_LOTE = 256


def _chunk_desde(texto: str, meta: dict[str, Any]) -> Chunk:
    return Chunk(
        document_id=meta["document_id"],
        bank=meta["bank"],
        url=meta["url"],
        title=meta.get("title", ""),
        heading=meta.get("heading", ""),
        text=texto,
        position=int(meta.get("position", 0)),
    )


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        *,
        max_retries: int = 3,
        client: Any = None,
    ) -> None:
        if client is None:
            import chromadb  # import perezoso: solo hace falta con un servidor real
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.HttpClient(
                host=host, port=port, settings=ChromaSettings(anonymized_telemetry=False)
            )
        self._client = client
        self._name = collection
        self._collection_cached: Any = None
        self._reintentar = retry(
            max_retries=max_retries, base_delay=1.0, retry_on=(VectorStoreError,)
        )

    @property
    def _collection(self) -> Any:
        if self._collection_cached is None:
            try:
                self._collection_cached = self._client.get_or_create_collection(
                    name=self._name, metadata={"hnsw:space": "cosine"}
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"No se pudo abrir la colección {self._name!r}: {exc}", cause=exc
                ) from exc
        return self._collection_cached

    def _ejecutar(self, descripcion: str, operacion: Callable[[], T]) -> T:
        """Ejecuta una operación contra Chroma traduciendo cualquier fallo a un error de dominio."""

        def intento() -> T:
            try:
                return operacion()
            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreError(f"{descripcion}: {exc}", cause=exc) from exc

        return self._reintentar(intento)()

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Cantidad de chunks y embeddings distinta")
        for i in range(0, len(chunks), _LOTE):
            lote = chunks[i : i + _LOTE]
            vectores = [list(e) for e in embeddings[i : i + _LOTE]]
            self._ejecutar(
                "Fallo al indexar chunks",
                lambda lote=lote, vectores=vectores: self._collection.upsert(
                    ids=[c.id for c in lote],
                    embeddings=vectores,
                    documents=[c.text for c in lote],
                    metadatas=[c.metadata() for c in lote],
                ),
            )

    def search(
        self, query_embedding: Sequence[float], top_k: int, bank: str | None = None
    ) -> list[RetrievedChunk]:
        r = self._ejecutar(
            "Fallo en la búsqueda vectorial",
            lambda: self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k,
                where={"bank": bank} if bank else None,
                include=["documents", "metadatas", "distances"],
            ),
        )
        resultados = []
        for texto, meta, distancia in zip(
            r["documents"][0], r["metadatas"][0], r["distances"][0], strict=True
        ):
            similitud = max(0.0, min(1.0, 1.0 - float(distancia)))  # distancia coseno
            resultados.append(
                RetrievedChunk(
                    _chunk_desde(texto, meta), similitud, RetrievalSource.VECTOR, similitud
                )
            )
        return resultados

    def list_chunks(self, bank: str | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        desplazamiento = 0
        while True:
            r = self._ejecutar(
                "Fallo al listar chunks",
                lambda d=desplazamiento: self._collection.get(
                    where={"bank": bank} if bank else None,
                    include=["documents", "metadatas"],
                    limit=_LOTE,
                    offset=d,
                ),
            )
            if not r["ids"]:
                return chunks
            chunks.extend(
                _chunk_desde(t, m) for t, m in zip(r["documents"], r["metadatas"], strict=True)
            )
            desplazamiento += len(r["ids"])

    def delete_bank(self, bank: str) -> None:
        self._ejecutar(
            f"Fallo al borrar el banco {bank!r}",
            lambda: self._collection.delete(where={"bank": bank}),
        )

    def count(self) -> int:
        return int(self._ejecutar("Fallo al contar chunks", lambda: self._collection.count()))

    def ping(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False
