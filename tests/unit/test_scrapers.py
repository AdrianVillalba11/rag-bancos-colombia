import pytest

from rag.config import Settings
from rag.domain.errors import HttpStatusError, ScrapingBlockedError, UnknownBankError
from rag.infra.http import FetchResult
from rag.ingestion.scrapers import ScraperFactory
from rag.ingestion.scrapers.base import BaseScraper

UA = "rag-bancos-colombia/0.1 (proyecto academico)"


class FakeFetcher:
    """Sitio simulado: url -> (status, texto, content-type)."""

    def __init__(self, sitio):
        self.sitio = sitio
        self.visitadas = []

    def get(self, url):
        self.visitadas.append(url)
        status, texto, tipo = self.sitio.get(url, (404, "", "text/html"))
        if status in (401, 403):
            raise ScrapingBlockedError(status, url)
        if status >= 400:
            raise HttpStatusError(status, url)
        return FetchResult(url, url, status, texto, tipo)


def html(titulo, *enlaces):
    cuerpo = "".join(f'<a href="{e}">x</a>' for e in enlaces)
    return (200, f"<html><title>{titulo}</title><body>{cuerpo}</body></html>", "text/html")


class BancoDePrueba(BaseScraper):
    bank = "prueba"
    base_url = "https://www.prueba.com"
    sitemap_urls = ("https://www.prueba.com/sitemap.xml",)
    include_prefixes = ("/personas",)
    start_paths = ("/personas",)


def scraper(sitio, **kw):
    fetcher = FakeFetcher(sitio)
    return BancoDePrueba(fetcher, user_agent=UA, **kw), fetcher


ROBOTS = (200, "User-agent: *\nDisallow: /personas/privado\nDisallow: /*pdf*\n", "text/plain")
SIN_ROBOTS = (404, "", "text/html")


def sitemap(*rutas):
    urls = "".join(f"<url><loc>https://www.prueba.com{r}</loc></url>" for r in rutas)
    return (200, f"<urlset>{urls}</urlset>", "text/xml")


def test_scrapea_desde_el_sitemap_respetando_robots_y_filtros():
    sitio = {
        "https://www.prueba.com/robots.txt": ROBOTS,
        "https://www.prueba.com/sitemap.xml": sitemap(
            "/personas/a", "/personas/privado/x", "/personas/guia-pdf", "/empresas/b", "/personas/c"
        ),
        "https://www.prueba.com/personas/a": html("A"),
        "https://www.prueba.com/personas/c": html("C"),
    }
    s, _ = scraper(sitio)
    paginas = list(s.scrape())
    assert sorted(p.url for p in paginas) == [
        "https://www.prueba.com/personas/a",
        "https://www.prueba.com/personas/c",
    ]
    assert all(p.bank == "prueba" for p in paginas)


def test_respeta_el_tope_de_paginas_y_omite_fallos_puntuales():
    sitio = {
        "https://www.prueba.com/robots.txt": SIN_ROBOTS,
        "https://www.prueba.com/sitemap.xml": sitemap(*[f"/personas/p{i}" for i in range(10)]),
        **{f"https://www.prueba.com/personas/p{i}": html(f"P{i}") for i in range(2, 10)},
    }  # p0 y p1 devuelven 404
    s, _ = scraper(sitio, max_pages=3)
    assert len(list(s.scrape())) == 3


def test_rastrea_por_enlaces_cuando_no_hay_sitemap():
    sitio = {
        "https://www.prueba.com/robots.txt": ROBOTS,
        "https://www.prueba.com/sitemap.xml": (200, "<urlset></urlset>", "text/xml"),
        "https://www.prueba.com/personas": html(
            "Inicio", "/personas/a", "/empresas/z", "/personas/privado/x"
        ),
        "https://www.prueba.com/personas/a": html("A", "/personas/a/b"),
        "https://www.prueba.com/personas/a/b": html("B"),
    }
    s, _ = scraper(sitio, max_depth=1)
    urls = [p.url for p in s.scrape()]
    assert urls == ["https://www.prueba.com/personas", "https://www.prueba.com/personas/a"]


def test_omite_paginas_duplicadas_por_contenido():
    sitio = {
        "https://www.prueba.com/robots.txt": SIN_ROBOTS,
        "https://www.prueba.com/sitemap.xml": sitemap("/personas/a", "/personas/b"),
        "https://www.prueba.com/personas/a": html("Igual"),
        "https://www.prueba.com/personas/b": html("Igual"),
    }
    s, _ = scraper(sitio)
    assert len(list(s.scrape())) == 1


def test_un_bloqueo_detiene_el_scraping_del_banco():
    sitio = {"https://www.prueba.com/robots.txt": (403, "", "text/html")}
    s, fetcher = scraper(sitio)
    with pytest.raises(ScrapingBlockedError):
        list(s.scrape())
    assert fetcher.visitadas == ["https://www.prueba.com/robots.txt"]  # no sigue insistiendo


def test_factory_crea_scrapers_y_rechaza_bancos_desconocidos():
    settings = Settings()
    assert ScraperFactory.available() == ["bancolombia", "bbva", "davivienda"]
    assert ScraperFactory.create("Bancolombia", FakeFetcher({}), settings).bank == "bancolombia"
    with pytest.raises(UnknownBankError):
        ScraperFactory.create("inexistente", FakeFetcher({}), settings)
