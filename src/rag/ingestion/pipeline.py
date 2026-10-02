"""Pipeline de ingesta: scraping -> limpieza -> chunking (-> indexación).

Cada etapa es independiente y se puede ejecutar por separado. Un banco que falla no detiene a
los demás: el error se registra y se reporta al final.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag.config import Settings
from rag.domain.errors import IngestionError, RagError
from rag.domain.models import Chunk
from rag.infra.http import HttpFetcher
from rag.ingestion.chunking import chunk_document
from rag.ingestion.cleaning import clean_corpus
from rag.ingestion.scrapers import ScraperFactory
from rag.ingestion.storage import CleanStore, RawStore

logger = logging.getLogger(__name__)

#: Recibe los chunks de un banco y los indexa (se conecta en la capa de recuperación)
Indexer = Callable[[str, list[Chunk]], None]


@dataclass(frozen=True)
class BankReport:
    bank: str
    raw_pages: int = 0
    documents: int = 0
    chunks: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def discarded(self) -> int:
        return self.raw_pages - self.documents


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        raw_store: RawStore | None = None,
        clean_store: CleanStore | None = None,
        indexer: Indexer | None = None,
    ) -> None:
        self._settings = settings
        self._raw = raw_store or RawStore(settings.raw_dir)
        self._clean = clean_store or CleanStore(settings.clean_dir)
        self._indexer = indexer

    # --- Etapa 1: scraping --------------------------------------------------------------------
    def scrape(self, banks: Sequence[str]) -> dict[str, str | None]:
        """Descarga los bancos indicados. Devuelve, por banco, el error o None si fue bien."""
        s = self._settings
        fetcher = HttpFetcher(
            user_agent=s.scrape_user_agent,
            timeout=s.scrape_timeout_seconds,
            max_retries=s.scrape_max_retries,
            delay=s.scrape_delay_seconds,
        )
        resultados: dict[str, str | None] = {}
        try:
            for banco in banks:
                try:
                    scraper = ScraperFactory.create(banco, fetcher, s)
                    total = self._raw.save_all(banco, scraper.scrape())
                    logger.info("Banco descargado", extra={"bank": banco, "pages": total})
                    resultados[banco] = None
                except RagError as exc:
                    logger.error(
                        "Banco omitido en el scraping",
                        extra={"bank": banco, "code": exc.code, "error": str(exc)},
                    )
                    resultados[banco] = str(exc)
        finally:
            fetcher.close()
        return resultados

    # --- Etapas 2 y 3: limpieza y chunking ----------------------------------------------------
    def process_bank(self, bank: str) -> BankReport:
        paginas = list(self._raw.load(bank))
        if not paginas:
            raise IngestionError(f"No hay páginas crudas para {bank!r}; ejecuta el scraping antes")

        documentos = clean_corpus(paginas)
        s = self._settings
        chunks = [
            chunk
            for doc in documentos
            for chunk in chunk_document(doc, s.chunk_size, s.chunk_overlap)
        ]
        self._clean.save_documents(bank, documentos)
        self._clean.save_chunks(bank, chunks)
        if self._indexer is not None:
            self._indexer(bank, chunks)

        reporte = BankReport(bank, len(paginas), len(documentos), len(chunks))
        logger.info(
            "Banco procesado",
            extra={"bank": bank, "raw": reporte.raw_pages, "docs": reporte.documents,
                   "chunks": reporte.chunks},
        )  # fmt: skip
        return reporte

    def process(self, banks: Sequence[str]) -> list[BankReport]:
        reportes: list[BankReport] = []
        for banco in banks:
            try:
                reportes.append(self.process_bank(banco))
            except RagError as exc:
                logger.error(
                    "Banco omitido en el procesamiento",
                    extra={"bank": banco, "code": exc.code, "error": str(exc)},
                )
                reportes.append(BankReport(banco, error=str(exc)))
        return reportes
