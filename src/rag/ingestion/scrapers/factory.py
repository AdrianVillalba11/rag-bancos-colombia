"""Factory de scrapers: crea la estrategia adecuada a partir del identificador del banco."""

from __future__ import annotations

from rag.config import Settings
from rag.domain.errors import UnknownBankError
from rag.domain.interfaces import Scraper
from rag.infra.http import Fetcher
from rag.ingestion.scrapers.bancolombia import BancolombiaScraper
from rag.ingestion.scrapers.base import BaseScraper
from rag.ingestion.scrapers.bbva import BbvaScraper
from rag.ingestion.scrapers.davivienda import DaviviendaScraper


class ScraperFactory:
    _registry: dict[str, type[BaseScraper]] = {
        BbvaScraper.bank: BbvaScraper,
        BancolombiaScraper.bank: BancolombiaScraper,
        DaviviendaScraper.bank: DaviviendaScraper,
    }

    @classmethod
    def register(cls, scraper_cls: type[BaseScraper]) -> None:
        """Permite añadir un banco nuevo sin modificar la fábrica."""
        cls._registry[scraper_cls.bank] = scraper_cls

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, bank: str, fetcher: Fetcher, settings: Settings) -> Scraper:
        try:
            scraper_cls = cls._registry[bank.lower()]
        except KeyError as exc:
            raise UnknownBankError(
                f"Banco desconocido: {bank!r}. Disponibles: {', '.join(cls.available())}"
            ) from exc
        return scraper_cls(
            fetcher,
            user_agent=settings.scrape_user_agent,
            max_pages=settings.scrape_max_pages_per_bank,
            max_depth=settings.scrape_max_depth,
        )
