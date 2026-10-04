"""Limitador de peticiones por ventana deslizante, en memoria.

Cada clave (por ejemplo, la IP del cliente) puede hacer `limit` peticiones dentro de los últimos
`window` segundos. Es suficiente para un despliegue de un solo proceso; con varias réplicas habría
que mover el contador a un almacén compartido (por ejemplo, Redis), y así se documenta.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if limit < 1 or window <= 0:
            raise ValueError("limit y window deben ser positivos")
        self._limit = limit
        self._window = window
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def hit(self, key: str) -> float:
        """Registra una petición. Devuelve 0 si se permite, o los segundos que faltan para poder
        volver a intentarlo si se superó el límite (en cuyo caso no se registra)."""
        ahora = self._clock()
        with self._lock:
            marcas = self._hits.setdefault(key, deque())
            while marcas and ahora - marcas[0] >= self._window:
                marcas.popleft()
            if len(marcas) >= self._limit:
                return max(0.001, self._window - (ahora - marcas[0]))
            marcas.append(ahora)
            if len(self._hits) > self._max_keys:
                self._purge(ahora)
            return 0.0

    def _purge(self, ahora: float) -> None:
        """Acota la memoria: elimina claves vencidas y, si aún sobran, las menos recientes."""
        vigentes = {k: v for k, v in self._hits.items() if v and ahora - v[-1] < self._window}
        if len(vigentes) > self._max_keys:
            ordenadas = sorted(vigentes, key=lambda k: vigentes[k][-1])
            for clave in ordenadas[: len(vigentes) - self._max_keys]:
                del vigentes[clave]
        self._hits = vigentes
