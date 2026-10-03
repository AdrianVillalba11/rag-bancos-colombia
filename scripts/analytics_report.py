"""Informe de analítica sobre el histórico de conversaciones.

Recorre todo el histórico guardado en PostgreSQL y calcula las métricas de uso, calidad e impacto.

Uso (dentro de Docker):
    docker compose run --rm ingest python scripts/analytics_report.py
    docker compose run --rm ingest python scripts/analytics_report.py --days 7
    docker compose run --rm ingest python scripts/analytics_report.py --format json
    docker compose run --rm ingest python scripts/analytics_report.py --output data/informe.md
"""

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag.analytics.metrics import compute_report
from rag.analytics.report import render_markdown
from rag.api.dependencies import build_conversation_repository
from rag.config import get_settings
from rag.infra.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--days", type=int, help="Solo los últimos N días (por defecto, todo)")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    parser.add_argument("--output", help="Guarda el informe en un archivo en vez de imprimirlo")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    repositorio = build_conversation_repository(settings)
    try:
        desde = datetime.now(UTC) - timedelta(days=args.days) if args.days else None
        informe = compute_report(
            repositorio.iter_messages(),
            since=desde,
            manual_minutes=settings.analytics_manual_search_minutes,
        )
    finally:
        repositorio.close()

    texto = (
        json.dumps(informe, ensure_ascii=False, indent=2)
        if args.format == "json"
        else render_markdown(informe)
    )
    if args.output:
        Path(args.output).write_text(texto, encoding="utf-8")
        print(f"Informe guardado en {args.output}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        print(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
