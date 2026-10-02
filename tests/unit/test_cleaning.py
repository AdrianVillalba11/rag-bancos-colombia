from rag.domain.models import RawPage
from rag.ingestion.cleaning import clean_corpus, extract_document

HTML = """
<html><head><title>Cuenta de Ahorros | Davivienda</title><script>var x = 1;</script></head>
<body>
  <header><a href="/">Logo del sitio</a></header>
  <nav><ul><li>Personas</li><li>Empresas</li></ul></nav>
  <main>
    <h1>Cuenta de Ahorros</h1>
    <p>Abra su cuenta <strong>sin costo</strong> y administre su dinero.</p>
    <p>Abra su cuenta sin costo y administre su dinero.</p>
    <h2>Requisitos</h2>
    <ul><li>Ser mayor de edad</li><li>Tener cédula de ciudadanía</li></ul>
    <table><tr><th>Plan</th><th>Cuota</th></tr><tr><td>Básico</td><td>$0</td></tr></table>
    <span>angle-right-small</span>
    <div class="cookie-banner"><p>Aceptar cookies</p></div>
    <div hidden><p>Texto oculto</p></div>
    <button>Enviar</button>
  </main>
  <footer><p>Pie de página legal</p></footer>
</body></html>
"""


def _doc():
    return extract_document(RawPage("davivienda", "https://x.com/cuentas", HTML))


def test_extrae_titulo_sin_el_sufijo_del_banco():
    assert _doc().title == "Cuenta de Ahorros"


def test_descarta_navegacion_scripts_ocultos_y_ruido():
    texto = _doc().text
    for ausente in ("Logo del sitio", "Empresas", "var x", "cookies", "oculto", "Enviar", "Pie de"):
        assert ausente not in texto
    assert "angle-right-small" not in texto


def test_organiza_el_texto_en_secciones_con_encabezado():
    doc = _doc()
    assert [s.heading for s in doc.sections] == ["Cuenta de Ahorros", "Requisitos"]
    assert "sin costo" in doc.sections[0].text  # el <strong> se integra en el párrafo


def test_elimina_parrafos_repetidos_dentro_de_la_pagina():
    assert _doc().text.lower().count("abra su cuenta") == 1


def test_las_tablas_se_aplanan_por_filas():
    assert "Plan | Cuota" in _doc().text
    assert "Básico | $0" in _doc().text


def _pagina(i: int, cuerpo: str, titulo: int | None = None) -> RawPage:
    html = (
        f"<html><title>P{i}</title><body><main><h1>Página {i if titulo is None else titulo}</h1>"
        f"<p>Menú repetido en todas las páginas del sitio web.</p>{cuerpo}</main></body></html>"
    )
    return RawPage("b", f"https://x.com/{i}", html)


def test_el_corpus_elimina_texto_repetitivo_vacios_y_duplicados():
    unico = "Este producto financiero ofrece condiciones especiales para clientes nuevos. " * 4
    paginas = [_pagina(i, f"<p>{unico} Variante {i}.</p>") for i in range(12)]
    paginas.append(_pagina(100, ""))  # solo menú: sin contenido propio
    paginas.append(_pagina(101, f"<p>{unico} Variante 0.</p>", titulo=0))  # igual que la 0
    docs = clean_corpus(paginas, min_chars=100)

    urls = {d.url for d in docs}
    assert "https://x.com/100" not in urls  # sin contenido propio
    assert "https://x.com/101" not in urls  # duplicada de la 0
    assert len(docs) == 12
    assert all("Menú repetido" not in d.text for d in docs)
    assert all("Variante" in d.text for d in docs)
