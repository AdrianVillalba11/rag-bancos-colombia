"""Limpieza del HTML crudo.

La limpieza ocurre en dos etapas:

1. **Estructural** (`extract_document`): por página, descarta navegación, scripts y elementos
   ocultos, y organiza el texto en secciones (encabezado + párrafos).
2. **De corpus** (`clean_corpus`): por banco, elimina texto repetitivo que aparece en muchas
   páginas (menús, pies de página, avisos), descarta páginas sin contenido propio y deduplica
   las que quedan idénticas. Es la que resuelve las "plantillas" que renderiza JavaScript.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from rag.domain.errors import CleaningError
from rag.domain.models import CleanDocument, RawPage, Section, content_hash

logger = logging.getLogger(__name__)

_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_DESCARTAR = (
    "script", "style", "noscript", "svg", "iframe", "template", "canvas", "button", "select",
    "option", "nav", "footer", "form", "aside", "dialog", "picture", "video", "audio",
)  # fmt: skip
_INLINE = ("a", "span", "b", "strong", "em", "i", "u", "small", "sup", "sub", "label", "abbr")
_CLASES_RUIDO = re.compile(
    r"cookie|consent|breadcrumb|skip-?link|sr-only|visually-hidden|modal|popup|chatbot", re.I
)
_ESPACIOS = re.compile(r"\s+")
# Textos de interfaz que no aportan información: nombres de iconos, botones y marcas del CMS
_LINEA_RUIDO = re.compile(
    r"^(ok|star|image|imagen|document|documento|startfragment|endfragment|external-link|"
    r"user-check|finaliza acordeion|visor de contenido web|"
    r"(angle|arrow|ico|icon|chevron)[\w-]*|se agreg[oó] icono.*|banner .*|"
    r"conoc(e|er) m[aá]s\*?|leer m[aá]s|ver m[aá]s|ir a la app|card|<.*>|"
    r"modal\b.*|header css|google tag manager.*|[a-z0-9]+(-[a-z0-9]+)+)$",  # kebab-case: iconos
    re.IGNORECASE,
)
_TITULO_SUFIJO = re.compile(r"\s*[|\-–—]\s*(BBVA( Colombia)?|Bancolombia|Davivienda)\s*$", re.I)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _ESPACIOS.sub(" ", text).strip()


def _fingerprint(text: str) -> str:
    """Clave para comparar líneas ignorando mayúsculas, acentos y puntuación."""
    base = unicodedata.normalize("NFKD", text.lower())
    base = "".join(c for c in base if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", base).strip()


def _es_ruido(linea: str) -> bool:
    return len(_fingerprint(linea)) < 3 or bool(_LINEA_RUIDO.match(linea))


def _prune(root: Tag) -> None:
    for etiqueta in root.find_all(_DESCARTAR):
        etiqueta.decompose()
    # `header` suelto es la cabecera del sitio; los que van dentro de secciones son títulos
    for cabecera in root.find_all("header"):
        if cabecera.find_parent(["main", "article", "section"]) is None:
            cabecera.decompose()
    for oculto in root.find_all(
        lambda t: isinstance(t, Tag)
        and t.attrs is not None
        and (
            t.has_attr("hidden")
            or t.get("aria-hidden") == "true"
            or "display:none" in (t.get("style") or "").replace(" ", "")
        )
    ):
        oculto.decompose()
    for ruido in root.find_all(
        lambda t: isinstance(t, Tag)
        and t.attrs is not None
        and _CLASES_RUIDO.search(" ".join(t.get("class", [])) + " " + (t.get("id") or ""))
    ):
        ruido.decompose()


def _flatten_tables(root: Tag) -> None:
    """Convierte cada fila de tabla en una sola línea 'celda | celda'."""
    for fila in root.find_all("tr"):
        celdas = [normalize_text(c.get_text(" ", strip=True)) for c in fila.find_all(["th", "td"])]
        celdas = [c for c in celdas if c]
        fila.replace_with(NavigableString(" | ".join(celdas) if celdas else ""))


def extract_document(page: RawPage) -> CleanDocument:
    """Etapa estructural: HTML -> documento con secciones."""
    try:
        sopa = BeautifulSoup(page.html, "lxml")
    except Exception as exc:  # lxml/bs4 pueden fallar con HTML corrupto
        raise CleaningError(f"No se pudo interpretar {page.url}", cause=exc) from exc

    titulo = normalize_text(sopa.title.get_text(" ", strip=True)) if sopa.title else ""
    titulo = _TITULO_SUFIJO.sub("", titulo)

    raiz = sopa.find("main") or sopa.body or sopa
    _prune(raiz)
    _flatten_tables(raiz)
    for etiqueta in raiz.find_all(_INLINE):
        etiqueta.unwrap()
    raiz.smooth()

    secciones: list[Section] = []
    encabezado = ""
    parrafos: list[str] = []
    vistas: set[str] = set()  # un mismo texto no se repite dentro de la página

    def cerrar() -> None:
        if parrafos:
            secciones.append(Section(encabezado, "\n".join(parrafos)))

    for texto in raiz.find_all(string=True):
        if not isinstance(texto, NavigableString) or texto.parent is None:
            continue
        limpio = normalize_text(str(texto))
        if not limpio or _es_ruido(limpio):
            continue
        if texto.find_parent(_HEADINGS) is not None:
            cerrar()
            encabezado, parrafos = limpio, []
            if not titulo and texto.find_parent("h1") is not None:
                titulo = limpio
        else:
            huella = _fingerprint(limpio)
            if huella in vistas:
                continue
            vistas.add(huella)
            parrafos.append(limpio)
    cerrar()

    return CleanDocument(
        bank=page.bank,
        url=page.url,
        title=titulo,
        sections=tuple(secciones),
        fetched_at=page.fetched_at,
        source_hash=page.hash,
    )


def _drop_lines(doc: CleanDocument, ruido: set[str]) -> CleanDocument:
    secciones = []
    for sec in doc.sections:
        if _fingerprint(sec.heading) in ruido:
            continue
        lineas = [ln for ln in sec.text.split("\n") if _fingerprint(ln) not in ruido]
        if lineas:
            secciones.append(Section(sec.heading, "\n".join(lineas)))
    return CleanDocument(
        doc.bank, doc.url, doc.title, tuple(secciones), doc.fetched_at, doc.source_hash
    )


def find_boilerplate(docs: list[CleanDocument], threshold: float, min_docs: int) -> set[str]:
    """Líneas presentes en una fracción alta de las páginas del banco (menús, pies, avisos)."""
    if len(docs) < min_docs:
        return set()
    frecuencia: Counter[str] = Counter()
    for doc in docs:
        lineas = {_fingerprint(sec.heading) for sec in doc.sections}
        for sec in doc.sections:
            lineas.update(_fingerprint(ln) for ln in sec.text.split("\n"))
        lineas.discard("")
        frecuencia.update(lineas)
    minimo = threshold * len(docs)
    return {linea for linea, n in frecuencia.items() if n >= minimo}


def clean_corpus(
    pages: Iterable[RawPage],
    *,
    boilerplate_threshold: float = 0.3,
    min_docs_for_boilerplate: int = 10,
    min_chars: int = 200,
) -> list[CleanDocument]:
    """Limpia todas las páginas de UN banco y devuelve los documentos con contenido propio."""
    documentos: list[CleanDocument] = []
    for pagina in pages:
        try:
            documentos.append(extract_document(pagina))
        except CleaningError as exc:
            logger.warning(
                "Página omitida en la limpieza", extra={"url": pagina.url, "error": str(exc)}
            )

    ruido = find_boilerplate(documentos, boilerplate_threshold, min_docs_for_boilerplate)
    logger.info("Texto repetitivo detectado", extra={"lines": len(ruido), "docs": len(documentos)})

    resultado: list[CleanDocument] = []
    vistos: set[str] = set()
    for doc in documentos:
        doc = _drop_lines(doc, ruido)
        texto = doc.text
        if len(texto) < min_chars:
            logger.debug("Página sin contenido propio descartada", extra={"url": doc.url})
            continue
        huella = content_hash(_fingerprint(texto))
        if huella in vistos:
            logger.debug("Página duplicada descartada", extra={"url": doc.url})
            continue
        vistos.add(huella)
        resultado.append(doc)
    return resultado
