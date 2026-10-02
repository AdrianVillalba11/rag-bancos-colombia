"""Lectura de sitemaps XML (índices y listados de URLs)."""

from __future__ import annotations

import logging
from xml.etree import ElementTree

from rag.domain.errors import ScrapingBlockedError, ScrapingError
from rag.infra.http import Fetcher

logger = logging.getLogger(__name__)


def parse_sitemap(xml_text: str) -> tuple[str, list[str]]:
    """Devuelve ("index" | "urlset", lista de ubicaciones <loc>)."""
    try:
        raiz = ElementTree.fromstring(xml_text.strip())
    except ElementTree.ParseError as exc:
        raise ScrapingError(f"Sitemap con XML inválido: {exc}", cause=exc) from exc

    def local(etiqueta: str) -> str:
        return etiqueta.rsplit("}", 1)[-1]

    tipo = "index" if local(raiz.tag) == "sitemapindex" else "urlset"
    ubicaciones = [
        (el.text or "").strip()
        for el in raiz.iter()
        if local(el.tag) == "loc" and (el.text or "").strip()
    ]
    return tipo, ubicaciones


def collect_sitemap_urls(
    fetcher: Fetcher, sitemap_url: str, *, max_sitemaps: int = 25, max_urls: int = 5000
) -> list[str]:
    """Recorre un sitemap (y sus sitemaps hijos) tolerando fallos en los hijos."""
    pendientes = [sitemap_url]
    visitados: set[str] = set()
    urls: list[str] = []

    while pendientes and len(visitados) < max_sitemaps and len(urls) < max_urls:
        actual = pendientes.pop(0)
        if actual in visitados:
            continue
        visitados.add(actual)
        try:
            respuesta = fetcher.get(actual)
            tipo, ubicaciones = parse_sitemap(respuesta.text)
        except ScrapingBlockedError:
            raise
        except ScrapingError as exc:
            logger.warning(
                "No se pudo leer el sitemap", extra={"sitemap": actual, "error": str(exc)}
            )
            continue
        if tipo == "index":
            pendientes.extend(ubicaciones)
        else:
            urls.extend(ubicaciones)
    return urls[:max_urls]
