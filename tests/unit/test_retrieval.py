import httpx
import pytest

from rag.domain.errors import EmbeddingError, RerankerError, RetrievalError
from rag.domain.models import Chunk, RetrievalSource, RetrievedChunk
from rag.retrieval.embeddings import OllamaEmbedder
from rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag.retrieval.indexer import ChunkIndexer
from rag.retrieval.lexical import Bm25Index, tokenize
from rag.retrieval.rerankers.base import NoopReranker
from rag.retrieval.rerankers.bge import BgeReranker


def chunk(texto: str, banco: str = "bbva", pos: int = 0, titulo: str = "T") -> Chunk:
    return Chunk(
        document_id=f"d-{texto[:5]}", bank=banco, url=f"https://x.com/{pos}", title=titulo,
        heading="", text=texto, position=pos,
    )  # fmt: skip


def rc(c: Chunk, score: float = 1.0, sim: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(c, score, RetrievalSource.VECTOR, sim)


# --- BM25 -------------------------------------------------------------------------------------
CORPUS = [
    chunk("El gravamen 4x1000 se aplica a los movimientos financieros", pos=1),
    chunk("Abre tu cuenta de ahorros sin cuota de manejo", pos=2),
    chunk("Conoce las tasas del CDT a término fijo", banco="davivienda", pos=3),
]


def test_tokenize_quita_acentos_y_palabras_vacias():
    assert tokenize("¿Cuál es la tasa del CDT?") == ["tasa", "cdt"]


def test_bm25_encuentra_terminos_exactos_y_siglas():
    indice = Bm25Index(CORPUS)
    assert indice.search("4x1000", 3)[0].chunk.position == 1
    assert indice.search("CDT", 3)[0].chunk.bank == "davivienda"


def test_bm25_filtra_por_banco_y_omite_resultados_sin_coincidencia():
    indice = Bm25Index(CORPUS)
    assert indice.search("CDT", 3, bank="bbva") == []
    assert indice.search("palabrainexistente", 3) == []
    assert Bm25Index([]).search("cualquiera", 3) == []


# --- RRF e híbrida ----------------------------------------------------------------------------
def test_rrf_premia_lo_que_aparece_en_ambas_listas():
    a, b, c = chunk("aaaaa", pos=1), chunk("bbbbb", pos=2), chunk("ccccc", pos=3)
    fusion = reciprocal_rank_fusion([[rc(a), rc(b)], [rc(c), rc(b)]], k=60)
    assert fusion[0].chunk.position == 2  # b está en las dos
    assert all(r.source == RetrievalSource.HYBRID for r in fusion)


def test_rrf_conserva_la_similitud_vectorial_y_respeta_top_k():
    a, b = chunk("aaaaa", pos=1), chunk("bbbbb", pos=2)
    fusion = reciprocal_rank_fusion([[rc(a, sim=0.8)], [rc(a), rc(b)]], top_k=1)
    assert len(fusion) == 1 and fusion[0].similarity == 0.8


class EmbedderFalso:
    def embed_query(self, texto):
        return [0.1, 0.2]

    def embed_documents(self, textos):
        return [[0.1, 0.2] for _ in textos]


class StoreFalso:
    def __init__(self, resultados=None, falla=False):
        self.resultados, self.falla, self.upserts, self.borrados = resultados or [], falla, [], []

    def search(self, vector, top_k, bank=None):
        if self.falla:
            raise RuntimeError("chroma caído")
        return self.resultados[:top_k]

    def upsert(self, chunks, embeddings):
        self.upserts.append((list(chunks), list(embeddings)))

    def delete_bank(self, bank):
        self.borrados.append(bank)


def test_recuperador_hibrido_combina_vectorial_y_lexico():
    vectorial = [rc(CORPUS[1], sim=0.7)]
    r = HybridRetriever(EmbedderFalso(), StoreFalso(vectorial), Bm25Index(CORPUS), top_k=5)
    posiciones = [x.chunk.position for x in r.retrieve("4x1000 cuenta")]
    assert 1 in posiciones and 2 in posiciones  # uno viene de BM25 y otro del vectorial


def test_sin_hibrida_solo_usa_la_busqueda_vectorial():
    vectorial = [rc(CORPUS[1], sim=0.7)]
    r = HybridRetriever(EmbedderFalso(), StoreFalso(vectorial), Bm25Index(CORPUS), hybrid=False)
    assert [x.chunk.position for x in r.retrieve("4x1000")] == [2]


def test_un_fallo_de_la_base_vectorial_se_traduce_a_error_de_dominio():
    r = HybridRetriever(EmbedderFalso(), StoreFalso(falla=True), None)
    with pytest.raises(RetrievalError):
        r.retrieve("hola")


# --- Reranker ---------------------------------------------------------------------------------
class ModeloFalso:
    def predict(self, pares):
        return [0.99 if "relevante" in doc else 0.01 for _, doc in pares]


def test_el_reranker_reordena_por_relevancia_y_conserva_el_puntaje():
    candidatos = [rc(chunk("ruido sin relación", pos=1)), rc(chunk("texto relevante", pos=2))]
    reranker = BgeReranker("m", model_loader=lambda: ModeloFalso())
    resultado = reranker.rerank("pregunta", candidatos, top_n=1)
    assert [r.chunk.position for r in resultado] == [2]
    assert resultado[0].source == RetrievalSource.RERANKED
    assert resultado[0].score == 0.99


def test_reranker_sin_candidatos_y_errores_de_dominio():
    assert BgeReranker("m", model_loader=lambda: ModeloFalso()).rerank("q", [], 3) == []

    def falla():
        raise OSError("sin modelo")

    with pytest.raises(RerankerError):
        BgeReranker("m", model_loader=falla).rerank("q", [rc(chunk("textoo"))], 1)


def test_noop_conserva_el_orden_y_recorta():
    cands = [rc(chunk(f"texto{i}", pos=i)) for i in range(4)]
    assert [r.chunk.position for r in NoopReranker().rerank("q", cands, 2)] == [0, 1]


# --- Embeddings (Ollama simulado) -------------------------------------------------------------
def _embedder(handler, **kw):
    cliente = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaEmbedder("http://ollama", "bge-m3", client=cliente, sleep=lambda _: None, **kw)


def test_embeddings_en_lotes_y_orden_preservado():
    lotes = []

    def handler(request):
        import json

        entrada = json.loads(request.content)["input"]
        lotes.append(len(entrada))
        return httpx.Response(200, json={"embeddings": [[float(len(t))] for t in entrada]})

    textos = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectores = _embedder(handler, batch_size=2).embed_documents(textos)
    assert lotes == [2, 2, 1]
    assert vectores == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_embeddings_reintenta_errores_5xx_y_no_los_4xx():
    intentos = []

    def handler(request):
        intentos.append(1)
        return httpx.Response(503 if len(intentos) < 3 else 200, json={"embeddings": [[1.0]]})

    assert _embedder(handler, max_retries=3).embed_query("hola") == [1.0]
    assert len(intentos) == 3

    def handler404(request):
        return httpx.Response(404, text="modelo no encontrado")

    with pytest.raises(EmbeddingError):
        _embedder(handler404, max_retries=3).embed_query("hola")


# --- Indexador --------------------------------------------------------------------------------
def test_el_indexador_reemplaza_el_contenido_del_banco():
    store = StoreFalso()
    ChunkIndexer(EmbedderFalso(), store)("bbva", CORPUS[:2])
    assert store.borrados == ["bbva"]
    assert len(store.upserts[0][0]) == 2 and len(store.upserts[0][1]) == 2


def test_el_indexador_ignora_listas_vacias():
    store = StoreFalso()
    ChunkIndexer(EmbedderFalso(), store)("bbva", [])
    assert store.borrados == [] and store.upserts == []
