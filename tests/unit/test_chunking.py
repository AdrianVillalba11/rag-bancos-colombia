from datetime import UTC, datetime

import pytest

from rag.domain.models import CleanDocument, Section
from rag.ingestion.chunking import chunk_document


def _doc(*secciones: Section) -> CleanDocument:
    return CleanDocument(
        bank="b", url="https://x.com/p", title="Producto", sections=tuple(secciones),
        fetched_at=datetime.now(UTC), source_hash="h",
    )  # fmt: skip


def _parrafos(n: int, largo: int = 150, prefijo: str = "p") -> str:
    return "\n".join(f"{prefijo}{i} " + "x" * (largo - 4) for i in range(n))


def test_un_documento_corto_es_un_solo_chunk_con_su_contexto():
    chunks = chunk_document(_doc(Section("Requisitos", "Ser mayor de edad.")), 800, 100)
    assert len(chunks) == 1
    assert chunks[0].heading == "Requisitos"
    assert chunks[0].embedding_text.startswith("Producto — Requisitos")


def test_ningun_chunk_supera_el_tamano_maximo():
    chunks = chunk_document(_doc(Section("T", _parrafos(30))), 400, 60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)


def test_los_parrafos_largos_se_dividen_por_frases():
    largo = ". ".join(f"Frase número {i} del párrafo extenso" for i in range(60)) + "."
    chunks = chunk_document(_doc(Section("T", largo)), 300, 40)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)


def test_hay_solape_entre_chunks_contiguos_de_una_seccion():
    chunks = chunk_document(_doc(Section("T", _parrafos(12, 100))), 450, 120)
    assert len(chunks) >= 3
    ultimas = chunks[0].text.split("\n")[-1]
    assert ultimas in chunks[1].text


def test_un_cambio_de_tema_cierra_el_chunk_sin_solape():
    a = Section("Tema A", _parrafos(4, 150, "a"))
    b = Section("Tema B", _parrafos(2, 150, "b"))
    chunks = chunk_document(_doc(a, b), 800, 100)
    headings = [c.heading for c in chunks]
    assert "Tema A" in headings and "Tema B" in headings
    assert all("a0" not in c.text for c in chunks if c.heading == "Tema B")


def test_los_temas_cortos_se_agrupan_en_un_chunk():
    doc = _doc(Section("A", "Texto breve."), Section("B", "Otro breve."))
    chunks = chunk_document(doc, 800, 100)
    assert len(chunks) == 1
    assert "A" in chunks[0].text and "B" in chunks[0].text


def test_los_ids_son_estables_y_las_posiciones_consecutivas():
    doc = _doc(Section("T", _parrafos(20)))
    a, b = chunk_document(doc, 400, 60), chunk_document(doc, 400, 60)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.position for c in a] == list(range(len(a)))
    assert len({c.id for c in a}) == len(a)


def test_valida_el_solape():
    with pytest.raises(ValueError):
        chunk_document(_doc(Section("T", "x")), 100, 100)
