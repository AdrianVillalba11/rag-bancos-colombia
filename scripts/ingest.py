"""Ejecuta la ingesta: (scraping opcional) -> limpieza -> chunking -> indexación.

Uso (dentro de Docker):
    docker compose run --rm ingest                        # procesa e indexa los datos crudos
    docker compose run --rm ingest python scripts/ingest.py --scrape      # descarga primero
    docker compose run --rm ingest python scripts/ingest.py --banks bbva  # un banco
    docker compose run --rm ingest python scripts/ingest.py --no-index    # sin indexar
"""

import argparse
import logging
import sys

from rag.config import get_settings
from rag.infra.logging import configure_logging
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.indexer import build_indexer

logger = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--banks", nargs="+", help="Bancos a procesar (por defecto, SCRAPE_BANKS)")
    parser.add_argument(
        "--scrape", action="store_true", help="Descarga las páginas antes de procesar"
    )
    parser.add_argument(
        "--no-index", action="store_true", help="Limpia y genera chunks sin indexar en ChromaDB"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    bancos = args.banks or settings.scrape_banks
    indexer = None if args.no_index else build_indexer(settings)
    pipeline = IngestionPipeline(settings, indexer=indexer)

    if args.scrape:
        pipeline.scrape(bancos)

    reportes = pipeline.process(bancos)
    print(f"\n{'banco':<14}{'crudas':>8}{'docs':>7}{'descartadas':>13}{'chunks':>8}")
    for r in reportes:
        if r.ok:
            print(f"{r.bank:<14}{r.raw_pages:>8}{r.documents:>7}{r.discarded:>13}{r.chunks:>8}")
        else:
            print(f"{r.bank:<14}ERROR: {r.error}")
    return 0 if any(r.ok for r in reportes) else 1


if __name__ == "__main__":
    sys.exit(main())
