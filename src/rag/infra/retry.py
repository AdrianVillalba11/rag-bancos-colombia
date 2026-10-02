"""Reintentos con backoff exponencial y jitter."""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


def compute_delay(attempt: int, base_delay: float, max_delay: float, jitter: bool) -> float:
    """Espera antes del reintento número `attempt` (1 = primer reintento)."""
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return random.uniform(0, delay) if jitter else delay


def retry(
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    should_retry: Callable[[BaseException], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorador que reintenta una función ante ciertas excepciones.

    - `max_retries`: reintentos adicionales al primer intento (0 desactiva el reintento).
    - `retry_on`: tipos de excepción que disparan un reintento; el resto se propaga de inmediato.
    - `should_retry`: filtro opcional adicional (por ejemplo, para ignorar errores HTTP 4xx).
    - `sleep`: inyectable para probar sin esperas reales.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:
                    attempt += 1
                    if attempt > max_retries or (should_retry and not should_retry(exc)):
                        raise
                    delay = compute_delay(attempt, base_delay, max_delay, jitter)
                    logger.warning(
                        "Reintentando operación tras un fallo",
                        extra={
                            "operation": func.__qualname__,
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "delay_seconds": round(delay, 3),
                            "error": repr(exc),
                        },
                    )
                    sleep(delay)

        return wrapper

    return decorator
