from rag.ingestion.scrapers.base import BaseScraper


class BancolombiaScraper(BaseScraper):
    """Sitio de Personas de Bancolombia. Publica un sitemap dedicado a esa sección."""

    bank = "bancolombia"
    base_url = "https://www.bancolombia.com"
    sitemap_urls = ("https://www.bancolombia.com/sitemap-personas.xml",)
    start_paths = ("/personas",)
    include_prefixes = ("/personas",)
    # Simuladores y buscadores son páginas interactivas sin contenido informativo propio
    exclude_patterns = (r"(?i)simulador", r"(?i)buscador")
