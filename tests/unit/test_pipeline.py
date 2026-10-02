import pytest

from rag.config import Settings
from rag.domain.errors import IngestionError
from rag.domain.models import RawPage
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.storage import CleanStore, RawStore


def _pagina(i: int) -> RawPage:
    cuerpo = f"Condiciones del producto financiero número {i} con requisitos detallados. " * 6
    html = (
        f"<html><title>Producto {i}</title><body><main><h1>Producto {i}</h1>"
        f"<p>{cuerpo}</p></main></body></html>"
    )
    return RawPage("demo", f"https://x.com/p{i}", html)


@pytest.fixture
def pipeline(tmp_path):
    settings = Settings(data_dir=tmp_path, chunk_size=300, chunk_overlap=40)
    raw, clean = RawStore(settings.raw_dir), CleanStore(settings.clean_dir)
    raw.save_all("demo", iter([_pagina(i) for i in range(4)]))
    return IngestionPipeline(settings, raw, clean), clean


def test_procesa_un_banco_y_guarda_documentos_y_chunks(pipeline):
    pipe, clean = pipeline
    reporte = pipe.process_bank("demo")
    assert reporte.ok and reporte.raw_pages == 4 and reporte.documents == 4
    assert reporte.chunks == len(clean.load_chunks("demo")) > 4
    docs = clean.load_documents("demo")
    assert {d.url for d in docs} == {f"https://x.com/p{i}" for i in range(4)}
    assert clean.banks() == ["demo"]


def test_el_procesamiento_es_idempotente(pipeline):
    pipe, clean = pipeline
    pipe.process_bank("demo")
    primeros = [c.id for c in clean.load_chunks("demo")]
    pipe.process_bank("demo")
    assert [c.id for c in clean.load_chunks("demo")] == primeros


def test_invoca_al_indexador_con_los_chunks_del_banco(tmp_path):
    settings = Settings(data_dir=tmp_path)
    raw = RawStore(settings.raw_dir)
    raw.save_all("demo", iter([_pagina(0)]))
    recibido = {}
    pipe = IngestionPipeline(
        settings, raw, CleanStore(settings.clean_dir),
        indexer=lambda banco, chunks: recibido.update({banco: chunks}),
    )  # fmt: skip
    pipe.process_bank("demo")
    assert recibido["demo"] and all(c.bank == "demo" for c in recibido["demo"])


def test_un_banco_sin_datos_falla_con_error_de_dominio_sin_detener_a_los_demas(pipeline):
    pipe, _ = pipeline
    with pytest.raises(IngestionError):
        pipe.process_bank("inexistente")
    reportes = pipe.process(["inexistente", "demo"])
    assert [r.ok for r in reportes] == [False, True]
