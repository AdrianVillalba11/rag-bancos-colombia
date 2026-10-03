"""Cliente de generación sobre Ollama (chat con y sin streaming)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx

from rag.domain.errors import LLMResponseError, LLMUnavailableError
from rag.domain.interfaces import LLMClient
from rag.domain.models import Message

logger = logging.getLogger(__name__)


class OllamaLLM(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        num_ctx: int = 8192,
        temperature: float = 0.2,
        timeout: float = 120.0,
        max_retries: int = 3,
        keep_alive: str = "10m",
        max_tokens: int = 700,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self._num_ctx = num_ctx
        self._temperature = temperature
        self._max_retries = max_retries
        self._keep_alive = keep_alive
        self._max_tokens = max_tokens
        self._sleep = sleep
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    # --- API pública --------------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        texto = "".join(self._stream(messages, max_tokens, temperature)).strip()
        if not texto:
            raise LLMResponseError("El modelo devolvió una respuesta vacía")
        return texto

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        return self._stream(messages, None, None)

    def ping(self) -> bool:
        try:
            return self._client.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    # --- Implementación -----------------------------------------------------------------------
    def _payload(
        self, messages: Sequence[Message], max_tokens: int | None, temperature: float | None
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "stream": True,
            "keep_alive": self._keep_alive,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "options": {
                "num_ctx": self._num_ctx,
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
        }

    def _stream(
        self, messages: Sequence[Message], max_tokens: int | None, temperature: float | None
    ) -> Iterator[str]:
        """Abre la conexión con reintentos y luego entrega los fragmentos.

        Solo se reintenta antes de recibir el primer fragmento; un corte a mitad de respuesta se
        informa como error porque repetir la solicitud duplicaría texto ya mostrado.
        """
        payload = self._payload(messages, max_tokens, temperature)
        intento = 0
        while True:
            emitido = False
            try:
                with self._client.stream("POST", "/api/chat", json=payload) as respuesta:
                    self._validar(respuesta)
                    for fragmento in self._leer(respuesta):
                        emitido = True
                        yield fragmento
                    return
            except httpx.HTTPError as exc:
                if emitido:
                    raise LLMUnavailableError(f"Se cortó la respuesta del modelo: {exc}") from exc
                error: Exception = _ReintentableError(f"No se pudo contactar con Ollama: {exc}")
            except _ReintentableError as exc:
                error = exc

            intento += 1
            if intento > self._max_retries:
                raise LLMUnavailableError(str(error), cause=error) from error
            espera = min(10.0, 1.0 * 2 ** (intento - 1))
            logger.warning(
                "Reintentando conexión con el modelo",
                extra={"attempt": intento, "delay_seconds": espera, "error": str(error)},
            )
            self._sleep(espera)

    @staticmethod
    def _validar(respuesta: httpx.Response) -> None:
        if respuesta.status_code == 200:
            return
        respuesta.read()
        detalle = respuesta.text[:200]
        if respuesta.status_code >= 500 or respuesta.status_code == 429:
            raise _ReintentableError(f"Ollama respondió HTTP {respuesta.status_code}: {detalle}")
        if respuesta.status_code == 404:
            raise LLMResponseError(f"Modelo no disponible en Ollama: {detalle}")
        raise LLMResponseError(f"Ollama rechazó la solicitud (HTTP {respuesta.status_code})")

    @staticmethod
    def _leer(respuesta: httpx.Response) -> Iterator[str]:
        for linea in respuesta.iter_lines():
            if not linea:
                continue
            try:
                datos = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise LLMResponseError("Fragmento inválido recibido del modelo") from exc
            if "error" in datos:
                raise LLMResponseError(f"Error del modelo: {datos['error']}")
            fragmento = datos.get("message", {}).get("content", "")
            if fragmento:
                yield fragmento
            if datos.get("done"):
                return


class _ReintentableError(Exception):
    """Fallo transitorio al abrir la conexión con el modelo."""
