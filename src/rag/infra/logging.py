"""Logging estructurado.

En formato JSON cada línea es un objeto con `timestamp`, `level`, `logger`, `message`
y los campos extra pasados con `logger.info("msg", extra={...})`.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Atributos estándar de LogRecord que no se vuelcan como campos extra
_RESERVADOS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entrada = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for clave, valor in record.__dict__.items():
            if clave not in _RESERVADOS:
                entrada[clave] = valor
        if record.exc_info:
            entrada["exception"] = self.formatException(record.exc_info)
        return json.dumps(entrada, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configura el logger raíz. Es idempotente: puede llamarse varias veces."""
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    raiz = logging.getLogger()
    raiz.handlers.clear()
    raiz.addHandler(handler)
    raiz.setLevel(level.upper())
    # Evita ruido de librerías HTTP en nivel INFO
    for ruidoso in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
