import pytest

from rag.domain.errors import ScrapingError
from rag.ingestion.sitemap import parse_sitemap
from rag.ingestion.urls import diversify, is_html_candidate, normalize_url, resolve, same_site


def test_normaliza_fragmento_query_y_barra_final():
    assert normalize_url("HTTPS://WWW.X.com/Personas/?a=1#top") == "https://www.x.com/Personas"
    assert normalize_url("https://x.com/") == "https://x.com/"


def test_resolve_descarta_enlaces_no_navegables():
    assert resolve("https://x.com/a/", "b") == "https://x.com/a/b"
    assert resolve("https://x.com/a", "mailto:a@b.com") is None
    assert resolve("https://x.com/a", "javascript:void(0)") is None
    assert resolve("https://x.com/a", "#seccion") is None


def test_mismo_sitio_ignora_www():
    assert same_site("https://x.com/a", "https://www.x.com")
    assert not same_site("https://otro.com/a", "https://www.x.com")


def test_filtra_archivos_no_html():
    assert is_html_candidate("https://x.com/personas/cuentas")
    assert not is_html_candidate("https://x.com/doc/guia.PDF")


def test_diversify_reparte_entre_secciones():
    urls = [
        "https://x.com/personas/creditos/a",
        "https://x.com/personas/creditos/b",
        "https://x.com/personas/creditos/c",
        "https://x.com/personas/cuentas/a",
        "https://x.com/personas/seguros/a",
    ]
    primeras = diversify(urls)[:3]
    assert {u.split("/")[4] for u in primeras} == {"creditos", "cuentas", "seguros"}


def test_parse_sitemap_urlset_e_indice():
    urlset = (
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://x.com/a</loc></url><url><loc>https://x.com/b?p=1&amp;q=2</loc></url>"
        "</urlset>"
    )
    assert parse_sitemap(urlset) == ("urlset", ["https://x.com/a", "https://x.com/b?p=1&q=2"])
    indice = (
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<sitemap><loc>https://x.com/s1.xml</loc></sitemap></sitemapindex>"
    )
    assert parse_sitemap(indice) == ("index", ["https://x.com/s1.xml"])


def test_parse_sitemap_vacio_o_invalido_lanza_error_de_dominio():
    with pytest.raises(ScrapingError):
        parse_sitemap("")
