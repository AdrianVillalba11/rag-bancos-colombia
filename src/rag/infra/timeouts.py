"""Timeout para llamadas bloqueantes que no lo soportan de forma nativa.

Las llamadas HTTP usan los timeouts propios del cliente. Esto es para librerías síncronas
(por ejemplo, la inferencia de un modelo local) que podrían quedarse colgadas.

Limitación: Python no puede cancelar un hilo en ejecución; si se agota el tiempo, la llamada
sigue corriendo en segundo plano hasta terminar, pero el invocador recupera el control.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TypeVar

from rag.domain.errors import OperationTimeoutError

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="timeout")


def call_with_timeout(func: Callable[[], T], seconds: float, operation: str = "operación") -> T:
    future = _executor.submit(func)
    try:
        return future.result(timeout=seconds)
    except FutureTimeoutError as exc:
        raise OperationTimeoutError(f"{operation} superó el límite de {seconds:g} s") from exc
