"""Ejecuta la ingesta: (scraping opcional) -> limpieza -> chunking.

Uso:
    python scripts/ingest.py                 # procesa los datos crudos ya descargados
    python scripts/ingest.py --scrape        # descarga primero y luego procesa
    python scripts/ingest.py --banks bbva    # un banco concreto
"""

import argparse
import logging
import sys

from rag.config import get_settings
from rag.infra.logging import configure_logging
from rag.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--banks", nargs="+", help="Bancos a procesar (por defecto, SCRAPE_BANKS)")
    parser.add_argument(
        "--scrape", action="store_true", help="Descarga las páginas antes de procesar"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    bancos = args.banks or settings.scrape_banks
    pipeline = IngestionPipeline(settings)

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
