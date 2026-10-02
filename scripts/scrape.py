"""Descarga las páginas de los bancos configurados y las guarda en data/raw.

Uso:
    python scripts/scrape.py                      # bancos de SCRAPE_BANKS
    python scripts/scrape.py --banks davivienda   # un banco concreto
    python scripts/scrape.py --max-pages 20       # prueba rápida
"""

import argparse
import logging
import sys

from rag.config import get_settings
from rag.infra.logging import configure_logging
from rag.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger("scrape")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--banks", nargs="+", help="Bancos a scrapear (por defecto, SCRAPE_BANKS)")
    parser.add_argument("--max-pages", type=int, help="Tope de páginas por banco")
    args = parser.parse_args()

    settings = get_settings()
    if args.max_pages:
        settings = settings.model_copy(update={"scrape_max_pages_per_bank": args.max_pages})
    configure_logging(settings.log_level, settings.log_json)

    bancos = args.banks or settings.scrape_banks
    resultados = IngestionPipeline(settings).scrape(bancos)

    fallidos = [banco for banco, error in resultados.items() if error]
    if fallidos:
        logger.warning("Scraping terminado con bancos fallidos", extra={"banks": fallidos})
    return 1 if len(fallidos) == len(bancos) else 0


if __name__ == "__main__":
    sys.exit(main())
