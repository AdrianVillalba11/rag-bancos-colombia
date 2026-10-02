"""Pruebas contra un ChromaDB real. Se ejecutan dentro de Docker (docker compose run --rm tests)
y se omiten si el servidor no está disponible."""

import os
import uuid

import pytest

from rag.domain.models import Chunk
from rag.retrieval.vector_store import ChromaVectorStore

HOST = os.environ.get("CHROMA_HOST", "localhost")
PORT = int(os.environ.get("CHROMA_PORT", "8000"))


@pytest.fixture
def store():
    try:
        s = ChromaVectorStore(HOST, PORT, f"test_{uuid.uuid4().hex[:8]}", max_retries=0)
        if not s.ping():
            pytest.skip("ChromaDB no disponible")
    except Exception:  # pragma: no cover
        pytest.skip("ChromaDB no disponible")
    yield s
    s._client.delete_collection(s._name)


def _chunk(texto: str, banco: str, pos: int) -> Chunk:
    return Chunk(
        document_id=f"doc{pos}", bank=banco, url=f"https://x.com/{pos}", title="T",
        heading="H", text=texto, position=pos,
    )  # fmt: skip


def test_indexa_busca_filtra_y_borra_por_banco(store):
    chunks = [
        _chunk("cuenta de ahorros", "bbva", 1),
        _chunk("crédito de vivienda", "davivienda", 2),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert(chunks, embeddings)
    store.upsert(chunks, embeddings)  # idempotente: no duplica
    assert store.count() == 2

    mejor = store.search([0.9, 0.1, 0.0], top_k=2)[0]
    assert mejor.chunk.bank == "bbva" and mejor.similarity > 0.9

    solo_davi = store.search([0.9, 0.1, 0.0], top_k=2, bank="davivienda")
    assert [r.chunk.bank for r in solo_davi] == ["davivienda"]

    assert {c.url for c in store.list_chunks()} == {"https://x.com/1", "https://x.com/2"}
    assert [c.bank for c in store.list_chunks(bank="bbva")] == ["bbva"]

    store.delete_bank("bbva")
    assert store.count() == 1
