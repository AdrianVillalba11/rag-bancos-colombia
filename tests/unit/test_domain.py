from datetime import UTC, datetime

from rag.domain.errors import (
    LLMUnavailableError,
    RagError,
    RobotsDisallowedError,
    ScrapingError,
)
from rag.domain.models import Chunk, CleanDocument, RawPage, Section, content_hash


def _chunk(text="Tasa del 1%", position=0):
    return Chunk(
        document_id="doc1", bank="bbva", url="https://x/y", title="T", heading="H",
        text=text, position=position,
    )


def test_el_id_del_chunk_es_determinista():
    assert _chunk().id == _chunk().id
    assert _chunk().id != _chunk(text="otro").id
    assert _chunk().id != _chunk(position=1).id


def test_metadata_del_chunk_incluye_banco_y_url():
    meta = _chunk().metadata()
    assert meta["bank"] == "bbva"
    assert meta["url"] == "https://x/y"


def test_hash_de_pagina_depende_del_contenido():
    a = RawPage(bank="bbva", url="u", html="<p>a</p>")
    b = RawPage(bank="bbva", url="u", html="<p>b</p>")
    assert a.hash == content_hash("<p>a</p>")
    assert a.hash != b.hash


def test_documento_limpio_compone_su_texto_por_secciones():
    doc = CleanDocument(
        bank="bbva", url="u", title="t",
        sections=(Section("Requisitos", "Ser mayor de edad"), Section("", "Texto suelto")),
        fetched_at=datetime.now(UTC), source_hash="h",
    )
    assert "Requisitos\nSer mayor de edad" in doc.text
    assert doc.text.endswith("Texto suelto")
    assert doc.id == doc.id


def test_jerarquia_y_codigos_de_errores():
    assert issubclass(RobotsDisallowedError, ScrapingError)
    assert issubclass(ScrapingError, RagError)
    assert RobotsDisallowedError.retryable is False
    assert LLMUnavailableError.retryable is True
    assert LLMUnavailableError("sin servicio").code == "llm_unavailable"


def test_la_lista_de_bancos_se_lee_separada_por_comas(monkeypatch):
    from rag.config import Settings

    monkeypatch.setenv("SCRAPE_BANKS", "BBVA, bancolombia")
    assert Settings().scrape_banks == ["bbva", "bancolombia"]
