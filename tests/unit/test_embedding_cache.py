import numpy as np

from rag.domain.models import Chunk
from rag.retrieval.embedding_cache import EmbeddingCache, chunk_fingerprints
from rag.retrieval.indexer import ChunkIndexer


def chunk(texto, pos=0, titulo="T"):
    return Chunk("doc", "bbva", f"https://x.com/{pos}", titulo, "H", texto, pos)


CHUNKS = [chunk("cuenta de ahorros", 0), chunk("tarjeta de credito", 1)]
VECTORES = [[0.5, -0.25, 0.125], [1.0, 0.0, -1.0]]


def test_guarda_y_recupera_los_vectores(tmp_path):
    cache = EmbeddingCache(tmp_path, "bge-m3")
    assert cache.load("bbva", CHUNKS) is None  # sin archivo
    assert cache.save("bbva", CHUNKS, VECTORES)
    recuperados = cache.load("bbva", CHUNKS)
    assert np.allclose(recuperados, VECTORES, atol=1e-3)  # media precisión


def test_se_invalida_si_cambia_el_modelo(tmp_path):
    EmbeddingCache(tmp_path, "bge-m3").save("bbva", CHUNKS, VECTORES)
    assert EmbeddingCache(tmp_path, "otro-modelo").load("bbva", CHUNKS) is None


def test_se_invalida_si_cambia_el_texto_el_titulo_o_el_orden(tmp_path):
    cache = EmbeddingCache(tmp_path, "bge-m3")
    cache.save("bbva", CHUNKS, VECTORES)
    assert cache.load("bbva", [chunk("otro texto", 0), CHUNKS[1]]) is None
    assert cache.load("bbva", [chunk("cuenta de ahorros", 0, titulo="Otro"), CHUNKS[1]]) is None
    assert cache.load("bbva", list(reversed(CHUNKS))) is None
    assert cache.load("bbva", CHUNKS[:1]) is None


def test_un_archivo_corrupto_se_ignora(tmp_path):
    (tmp_path / "bbva").mkdir()
    (tmp_path / "bbva" / "embeddings.npz").write_bytes(b"no es un npz")
    assert EmbeddingCache(tmp_path, "bge-m3").load("bbva", CHUNKS) is None


def test_si_no_se_puede_escribir_no_falla(tmp_path):
    bloqueo = tmp_path / "bloqueado"
    bloqueo.write_text("soy un archivo, no un directorio")  # mkdir fallará
    assert EmbeddingCache(bloqueo, "bge-m3").save("bbva", CHUNKS, VECTORES) is False


def test_las_huellas_dependen_del_texto_que_se_vectoriza():
    assert chunk_fingerprints([chunk("a", titulo="X")]) != chunk_fingerprints(
        [chunk("a", titulo="Y")]
    )
    assert chunk_fingerprints(CHUNKS) == chunk_fingerprints(CHUNKS)


class EmbedderEspia:
    def __init__(self):
        self.llamadas = 0

    def embed_documents(self, textos):
        self.llamadas += 1
        return [[float(i), 0.0, 1.0] for i, _ in enumerate(textos)]

    def embed_query(self, texto):
        return [0.0, 0.0, 0.0]


class StoreFalso:
    def __init__(self):
        self.upserts = []

    def delete_bank(self, bank):
        pass

    def upsert(self, chunks, vectores):
        self.upserts.append(list(vectores))


def test_el_indexador_calcula_una_vez_y_luego_reutiliza_la_cache(tmp_path):
    cache = EmbeddingCache(tmp_path, "bge-m3")
    embedder, store = EmbedderEspia(), StoreFalso()
    indexador = ChunkIndexer(embedder, store, cache)

    indexador("bbva", CHUNKS)  # primera vez: calcula y guarda
    indexador("bbva", CHUNKS)  # segunda: reutiliza
    assert embedder.llamadas == 1
    assert np.allclose(store.upserts[0], store.upserts[1], atol=1e-3)


def test_el_indexador_recalcula_si_los_chunks_cambian(tmp_path):
    cache = EmbeddingCache(tmp_path, "bge-m3")
    embedder = EmbedderEspia()
    indexador = ChunkIndexer(embedder, StoreFalso(), cache)
    indexador("bbva", CHUNKS)
    indexador("bbva", [chunk("contenido nuevo", 0)])
    assert embedder.llamadas == 2


def test_el_indexador_sin_cache_siempre_calcula():
    embedder = EmbedderEspia()
    indexador = ChunkIndexer(embedder, StoreFalso())
    indexador("bbva", CHUNKS)
    indexador("bbva", CHUNKS)
    assert embedder.llamadas == 2
