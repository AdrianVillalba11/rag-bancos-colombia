"""Scraper base (patrón Template Method sobre la estrategia `Scraper`).

El flujo común es: leer robots.txt → descubrir URLs (sitemap o, si no hay, rastreo por enlaces)
→ descargar respetando robots, un tope de páginas y un ritmo cortés → producir `RawPage`.
Cada banco solo declara su configuración (URLs de inicio, sitemaps, filtros).
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Iterator
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from rag.domain.errors import HttpStatusError, ScrapingBlockedError, ScrapingError
from rag.domain.interfaces import Scraper
from rag.domain.models import RawPage
from rag.infra.http import Fetcher
from rag.ingestion import urls as url_utils
from rag.ingestion.robots import RobotsPolicy
from rag.ingestion.sitemap import collect_sitemap_urls

logger = logging.getLogger(__name__)


class BaseScraper(Scraper):
    base_url: str
    #: Sitemaps conocidos del sitio (si no se declara ninguno, se usan los del robots.txt)
    sitemap_urls: tuple[str, ...] = ()
    #: Permite usar los sitemaps que anuncia el robots.txt cuando no se declaran propios
    use_robots_sitemaps: bool = True
    #: Rutas de inicio del rastreo por enlaces, cuando no hay sitemap utilizable
    start_paths: tuple[str, ...] = ("/",)
    #: Solo se siguen URLs cuya ruta empiece por alguno de estos prefijos (vacío = todas)
    include_prefixes: tuple[str, ...] = ()
    #: Expresiones regulares sobre la ruta que se descartan
    exclude_patterns: tuple[str, ...] = ()

    def __init__(
        self,
        fetcher: Fetcher,
        *,
        user_agent: str,
        max_pages: int = 150,
        max_depth: int = 3,
    ) -> None:
        self._fetcher = fetcher
        self._user_agent = user_agent
        self._max_pages = max_pages
        self._max_depth = max_depth
        self._exclude = [re.compile(p) for p in self.exclude_patterns]

    # --- API pública --------------------------------------------------------------------------
    def scrape(self) -> Iterator[RawPage]:
        robots = self._load_robots()
        candidatas = self._discover_from_sitemaps(robots)
        if candidatas:
            logger.info(
                "Descubrimiento por sitemap", extra={"bank": self.bank, "urls": len(candidatas)}
            )
            fuente = self._fetch_list(candidatas)
        else:
            logger.info("Sin sitemap utilizable; se rastrea por enlaces", extra={"bank": self.bank})
            fuente = self._crawl(robots)

        vistos: set[str] = set()
        entregadas = 0
        for pagina in fuente:
            if pagina.hash in vistos:
                logger.info(
                    "Página duplicada omitida", extra={"bank": self.bank, "url": pagina.url}
                )
                continue
            vistos.add(pagina.hash)
            entregadas += 1
            yield pagina
        logger.info("Scraping finalizado", extra={"bank": self.bank, "pages": entregadas})

    # --- Robots y descubrimiento --------------------------------------------------------------
    def _load_robots(self) -> RobotsPolicy:
        url = self.base_url.rstrip("/") + "/robots.txt"
        try:
            respuesta = self._fetcher.get(url)
        except ScrapingBlockedError:
            raise
        except HttpStatusError as exc:
            if exc.status_code == 404:
                logger.info("El sitio no publica robots.txt", extra={"bank": self.bank})
                return RobotsPolicy.allow_all()
            raise
        return RobotsPolicy.parse(respuesta.text, self._user_agent)

    def _discover_from_sitemaps(self, robots: RobotsPolicy) -> list[str]:
        sitemaps = self.sitemap_urls or (tuple(robots.sitemaps) if self.use_robots_sitemaps else ())
        encontradas: list[str] = []
        for sitemap in sitemaps:
            encontradas.extend(collect_sitemap_urls(self._fetcher, sitemap))
        candidatas = {
            url_utils.normalize_url(u) for u in encontradas if self._is_candidate(u, robots)
        }
        return url_utils.diversify(sorted(candidatas))

    def _is_candidate(self, url: str, robots: RobotsPolicy) -> bool:
        if not url_utils.same_site(url, self.base_url) or not url_utils.is_html_candidate(url):
            return False
        ruta = urlsplit(url).path or "/"
        if self.include_prefixes and not ruta.startswith(self.include_prefixes):
            return False
        if any(p.search(ruta) for p in self._exclude):
            return False
        return robots.allowed(url)

    # --- Descarga -----------------------------------------------------------------------------
    def _fetch(self, url: str) -> RawPage | None:
        """Descarga una página. Los fallos puntuales se registran y no interrumpen el proceso."""
        try:
            respuesta = self._fetcher.get(url)
        except ScrapingBlockedError:
            raise
        except ScrapingError as exc:
            logger.warning(
                "Página omitida", extra={"bank": self.bank, "url": url, "error": str(exc)}
            )
            return None
        if not respuesta.is_html or not url_utils.same_site(respuesta.final_url, self.base_url):
            return None
        return RawPage(
            bank=self.bank,
            url=url,
            html=respuesta.text,
            status_code=respuesta.status_code,
        )

    def _fetch_list(self, urls: list[str]) -> Iterator[RawPage]:
        entregadas = 0
        for url in urls:
            if entregadas >= self._max_pages:
                return
            pagina = self._fetch(url)
            if pagina is not None:
                entregadas += 1
                yield pagina

    def _crawl(self, robots: RobotsPolicy) -> Iterator[RawPage]:
        """Rastreo en anchura desde las rutas de inicio, hasta `max_depth` y `max_pages`."""
        cola: deque[tuple[str, int]] = deque(
            (url_utils.normalize_url(self.base_url.rstrip("/") + p), 0) for p in self.start_paths
        )
        vistas = {url for url, _ in cola}
        entregadas = 0

        while cola and entregadas < self._max_pages:
            url, profundidad = cola.popleft()
            if not robots.allowed(url):
                continue
            pagina = self._fetch(url)
            if pagina is None:
                continue
            entregadas += 1
            yield pagina

            if profundidad >= self._max_depth:
                continue
            for enlace in self._extract_links(pagina):
                if enlace not in vistas and self._is_candidate(enlace, robots):
                    vistas.add(enlace)
                    cola.append((enlace, profundidad + 1))

    @staticmethod
    def _extract_links(pagina: RawPage) -> list[str]:
        sopa = BeautifulSoup(pagina.html, "lxml")
        enlaces = (url_utils.resolve(pagina.url, a["href"]) for a in sopa.find_all("a", href=True))
        return list(dict.fromkeys(e for e in enlaces if e))
