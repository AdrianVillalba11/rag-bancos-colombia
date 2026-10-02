from rag.ingestion.scrapers.base import BaseScraper


class DaviviendaScraper(BaseScraper):
    """Sitio de Davivienda (personas y páginas institucionales).

    Sus sitemaps hijos se publican vacíos, así que se rastrea por enlaces desde la portada.
    La sección /personas es pequeña, por lo que no se restringe el rastreo a ella; el
    robots.txt del sitio (que excluye /documents/, /wps/wcm/connect y las URLs con query)
    sigue acotando lo que se descarga.
    """

    bank = "davivienda"
    base_url = "https://www.davivienda.com"
    use_robots_sitemaps = False
    start_paths = ("/",)
    # Rutas de portales internos que no contienen contenido informativo
    exclude_patterns = (r"^/wps/",)
