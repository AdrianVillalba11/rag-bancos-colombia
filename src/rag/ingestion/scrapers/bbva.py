from rag.ingestion.scrapers.base import BaseScraper


class BbvaScraper(BaseScraper):
    """BBVA Colombia.

    Nota: el sitio está detrás de un WAF que responde HTTP 403 a ciertos clientes (por ejemplo,
    `curl` en Windows) aunque envíen el mismo User-Agent y cabeceras, y HTTP 200 a otros
    (`httpx`). Es un filtrado del lado del servidor que no controlamos. El scraper se
    identifica siempre con su propio User-Agent, respeta robots.txt y, si recibe un 401/403,
    se detiene con `ScrapingBlockedError` sin intentar evadir el bloqueo.
    """

    bank = "bbva"
    base_url = "https://www.bbva.com.co"
    sitemap_urls = ("https://www.bbva.com.co/sitemap.xml",)
    start_paths = ("/",)
    exclude_patterns = (r"(?i)buscador",)
