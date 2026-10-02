"""División de documentos en chunks respetando su estructura.

Se empaquetan párrafos completos hasta `chunk_size` caracteres. Los encabezados marcan temas:
al cambiar de sección se cierra el chunk (si ya tiene un mínimo de contenido) para no mezclar
temas distintos. Entre chunks contiguos de una misma sección se repite un solape de hasta
`chunk_overlap` caracteres, tomando párrafos o frases completas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.domain.models import Chunk, CleanDocument

_FRASES = re.compile(r"(?<=[.!?;:])\s+")
#: Un chunk con menos de esta fracción de `chunk_size` se sigue llenando aunque cambie el tema
_FRACCION_MINIMA = 0.35
_MIN_CHARS_CHUNK_FINAL = 120


@dataclass(frozen=True)
class _Unidad:
    heading: str
    text: str
    es_encabezado: bool = False


def _partir_texto_largo(texto: str, limite: int) -> list[str]:
    """Divide un párrafo demasiado largo por frases y, si hace falta, por palabras."""
    piezas: list[str] = []
    actual = ""
    for frase in _FRASES.split(texto):
        for trozo in _por_palabras(frase, limite):
            if actual and len(actual) + 1 + len(trozo) > limite:
                piezas.append(actual)
                actual = trozo
            else:
                actual = f"{actual} {trozo}".strip()
    if actual:
        piezas.append(actual)
    return piezas


def _por_palabras(texto: str, limite: int) -> list[str]:
    if len(texto) <= limite:
        return [texto]
    trozos, actual = [], ""
    for palabra in texto.split(" "):
        if actual and len(actual) + 1 + len(palabra) > limite:
            trozos.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}".strip()
    if actual:
        trozos.append(actual)
    return trozos


def _unidades(doc: CleanDocument, chunk_size: int) -> list[_Unidad]:
    unidades: list[_Unidad] = []
    for seccion in doc.sections:
        if seccion.heading:
            unidades.append(_Unidad(seccion.heading, seccion.heading, es_encabezado=True))
        for linea in seccion.text.split("\n"):
            partes = [linea] if len(linea) <= chunk_size else _partir_texto_largo(linea, chunk_size)
            unidades.extend(_Unidad(seccion.heading, p) for p in partes if p)
    return unidades


def _largo(unidades: list[_Unidad]) -> int:
    return sum(len(u.text) for u in unidades) + max(0, len(unidades) - 1)


def chunk_document(
    doc: CleanDocument, chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    grupos: list[list[_Unidad]] = []
    actual: list[_Unidad] = []

    def cerrar(solape: bool) -> None:
        nonlocal actual
        if not actual:
            return
        grupos.append(actual)
        cola: list[_Unidad] = []
        if solape:
            # Solape: últimas unidades (no encabezados) hasta `chunk_overlap` caracteres
            for u in reversed(actual):
                if u.es_encabezado or _largo([u, *cola]) > chunk_overlap:
                    break
                cola.insert(0, u)
        actual = cola

    for unidad in _unidades(doc, chunk_size):
        cambia_tema = unidad.es_encabezado and _largo(actual) >= chunk_size * _FRACCION_MINIMA
        if cambia_tema:
            cerrar(solape=False)
            actual = []
        elif actual and _largo([*actual, unidad]) > chunk_size:
            cerrar(solape=True)
            if actual and _largo([*actual, unidad]) > chunk_size:
                actual = []  # el solape no cabe junto a la unidad nueva
        actual.append(unidad)
    if actual and any(not u.es_encabezado for u in actual):
        grupos.append(actual)

    # Un último chunk diminuto se une al anterior (salvo que sea solo solape)
    if len(grupos) >= 2 and _largo(grupos[-1]) < _MIN_CHARS_CHUNK_FINAL:
        ultimo = grupos.pop()
        previo_textos = {u.text for u in grupos[-1]}
        grupos[-1].extend(u for u in ultimo if u.text not in previo_textos)

    chunks: list[Chunk] = []
    for posicion, grupo in enumerate(grupos):
        texto = "\n".join(u.text for u in grupo).strip()
        if not texto:
            continue
        chunks.append(
            Chunk(
                document_id=doc.id,
                bank=doc.bank,
                url=doc.url,
                title=doc.title,
                heading=grupo[0].heading,
                text=texto,
                position=posicion,
            )
        )
    return chunks
