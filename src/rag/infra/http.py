"""Cliente HTTP con timeouts, reintentos con backoff y espera cortés entre solicitudes."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from rag.domain.errors import HttpStatusError, ScrapingBlockedError, ScrapingError
from rag.infra.retry import retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    text: str
    content_type: str

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.lower()


class Fetcher(Protocol):
    """Contrato mínimo que necesitan los scrapers; permite sustituirlo en las pruebas."""

    def get(self, url: str) -> FetchResult: ...


class HttpFetcher:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 20.0,
        max_retries: int = 3,
        delay: float = 1.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Language": "es-CO,es;q=0.9"},
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )
        self._delay = delay
        self._sleep = sleep
        self._clock = clock
        self._last_request = 0.0
        self._get_with_retry = retry(
            max_retries=max_retries,
            base_delay=1.0,
            max_delay=15.0,
            retry_on=(ScrapingError,),
            should_retry=lambda exc: getattr(exc, "retryable", False),
            sleep=sleep,
        )(self._get_once)

    def get(self, url: str) -> FetchResult:
        return self._get_with_retry(url)

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        espera = self._delay - (self._clock() - self._last_request)
        if espera > 0:
            self._sleep(espera)
        self._last_request = self._clock()

    def _get_once(self, url: str) -> FetchResult:
        self._throttle()
        try:
            response = self._client.get(url)
        except httpx.TimeoutException as exc:
            raise ScrapingError(f"Tiempo de espera agotado al solicitar {url}", cause=exc) from exc
        except httpx.HTTPError as exc:
            raise ScrapingError(f"Error de red al solicitar {url}: {exc}", cause=exc) from exc

        status = response.status_code
        if status in (401, 403):
            raise ScrapingBlockedError(status, url)
        if status >= 400:
            raise HttpStatusError(status, url)
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=status,
            text=response.text,
            content_type=response.headers.get("content-type", ""),
        )
