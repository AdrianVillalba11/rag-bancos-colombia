from rag.ingestion.robots import RobotsPolicy

UA = "rag-bancos-colombia/0.1 (proyecto academico)"

ROBOTS = """
User-agent: GPTBot
Disallow: /

User-agent: *
User-agent: Googlebot
Disallow: /*pdf*
Disallow: /rest/
Disallow: /personas/solicitud-de-productos/
Allow: /personas/solicitud-de-productos/info
Disallow: /*.zip$
Sitemap: https://x.com/sitemap.xml
"""


def test_usa_el_grupo_comodin_y_ignora_los_de_otros_bots():
    robots = RobotsPolicy.parse(ROBOTS, UA)
    assert robots.allowed("https://x.com/personas/cuentas")
    assert robots.allowed("https://x.com/")


def test_comodines_y_prefijos():
    robots = RobotsPolicy.parse(ROBOTS, UA)
    assert not robots.allowed("https://x.com/personas/guia-pdf-ahorro")
    assert not robots.allowed("https://x.com/rest/api")
    assert not robots.allowed("https://x.com/archivos/datos.zip")
    assert robots.allowed("https://x.com/archivos/datos.zip.html")  # el $ ancla el final


def test_gana_la_regla_mas_especifica():
    robots = RobotsPolicy.parse(ROBOTS, UA)
    assert not robots.allowed("https://x.com/personas/solicitud-de-productos/tarjeta")
    assert robots.allowed("https://x.com/personas/solicitud-de-productos/info")


def test_un_grupo_especifico_tiene_prioridad_sobre_el_comodin():
    texto = "User-agent: *\nDisallow: /\n\nUser-agent: rag-bancos-colombia\nAllow: /\n"
    assert RobotsPolicy.parse(texto, UA).allowed("https://x.com/algo")


def test_la_query_se_evalua():
    robots = RobotsPolicy.parse("User-agent: *\nDisallow: /*?\n", UA)
    assert not robots.allowed("https://x.com/pagina?a=1")
    assert robots.allowed("https://x.com/pagina")


def test_lee_los_sitemaps_y_permite_todo_sin_reglas():
    assert RobotsPolicy.parse(ROBOTS, UA).sitemaps == ["https://x.com/sitemap.xml"]
    assert RobotsPolicy.allow_all().allowed("https://x.com/cualquiera")
