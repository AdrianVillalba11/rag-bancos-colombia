import httpx
import pytest

from rag.domain.errors import HttpStatusError, ScrapingBlockedError, ScrapingError
from rag.domain.models import RawPage
from rag.infra.http import HttpFetcher
from rag.ingestion.storage import RawStore


def test_guarda_y_recupera_paginas_crudas(tmp_path):
    store = RawStore(tmp_path)
    paginas = [RawPage("bbva", f"https://x.com/{i}", f"<p>ñandú {i}</p>") for i in range(3)]
    assert store.save_all("bbva", iter(paginas)) == 3
    assert store.banks() == ["bbva"]
    leidas = list(store.load("bbva"))
    assert [p.html for p in leidas] == [p.html for p in sorted(paginas, key=lambda p: p.url)]


def test_el_guardado_es_idempotente_y_limpia_paginas_obsoletas(tmp_path):
    store = RawStore(tmp_path)
    store.save_all(
        "b", iter([RawPage("b", "https://x.com/1", "uno"), RawPage("b", "https://x.com/2", "dos")])
    )
    store.save_all("b", iter([RawPage("b", "https://x.com/1", "uno")]))
    assert len(list((tmp_path / "b").glob("*.html.gz"))) == 1


def test_si_el_scraping_falla_sin_paginas_se_conserva_lo_anterior(tmp_path):
    store = RawStore(tmp_path)
    store.save_all("b", iter([RawPage("b", "https://x.com/1", "uno")]))

    def falla():
        raise ScrapingBlockedError(403, "https://x.com")
        yield  # pragma: no cover

    with pytest.raises(ScrapingBlockedError):
        store.save_all("b", falla())
    assert len(list(store.load("b"))) == 1


def _fetcher(handler, **kw):
    cliente = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return HttpFetcher(user_agent="t", client=cliente, delay=0, sleep=lambda _: None, **kw)


def test_reintenta_errores_transitorios_y_luego_tiene_exito():
    intentos = []

    def handler(request):
        intentos.append(1)
        if len(intentos) < 3:
            return httpx.Response(503)
        return httpx.Response(200, text="ok", headers={"content-type": "text/html; charset=utf-8"})

    resultado = _fetcher(handler, max_retries=3).get("https://x.com")
    assert resultado.text == "ok" and resultado.is_html
    assert len(intentos) == 3


def test_no_reintenta_bloqueos_ni_404():
    llamadas = []

    def handler(request):
        llamadas.append(request.url.path)
        return httpx.Response(403 if request.url.path == "/b" else 404)

    f = _fetcher(handler, max_retries=3)
    with pytest.raises(ScrapingBlockedError):
        f.get("https://x.com/b")
    with pytest.raises(HttpStatusError) as info:
        f.get("https://x.com/n")
    assert info.value.status_code == 404
    assert llamadas == ["/b", "/n"]


def test_errores_de_red_se_convierten_en_error_de_dominio():
    def handler(request):
        raise httpx.ConnectError("sin red")

    with pytest.raises(ScrapingError):
        _fetcher(handler, max_retries=1).get("https://x.com")
