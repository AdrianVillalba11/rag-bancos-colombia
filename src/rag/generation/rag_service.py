"""Servicio RAG (patrón Facade).

Oculta tras una sola llamada todo el flujo: reescritura de la pregunta con el historial,
recuperación híbrida, reranking, guardrail de relevancia, generación y citas. La API solo conoce
`RagService.stream_answer` / `answer`.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence

from rag.config import Settings
from rag.domain.errors import LLMResponseError, LLMUnavailableError, RagError, RerankerError
from rag.domain.interfaces import LLMClient, Reranker
from rag.domain.models import (
    AnswerMetrics,
    Citation,
    Message,
    RagAnswer,
    RetrievalSource,
    RetrievedChunk,
    StreamEvent,
)
from rag.generation import prompts
from rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

_CITA = re.compile(r"\[(\d{1,2})\]")
_MAX_FUENTES_SIN_CITA = 3


def _ms(inicio: float) -> int:
    return int((time.perf_counter() - inicio) * 1000)


class RagService:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._settings = settings

    # --- API pública --------------------------------------------------------------------------
    def answer(
        self, question: str, history: Sequence[Message] = (), bank: str | None = None
    ) -> RagAnswer:
        final: RagAnswer | None = None
        for evento in self.stream_answer(question, history, bank):
            if evento.final is not None:
                final = evento.final
        assert final is not None
        return final

    def stream_answer(
        self, question: str, history: Sequence[Message] = (), bank: str | None = None
    ) -> Iterator[StreamEvent]:
        inicio = time.perf_counter()
        history = list(history)[-self._settings.history_max_messages :] if history else []

        consulta = self._rewrite(question, history)
        t0 = time.perf_counter()
        candidatos = self._retriever.retrieve(consulta, bank)
        retrieval_ms = _ms(t0)

        t0 = time.perf_counter()
        elegidos = self._rerank(consulta, candidatos)
        rerank_ms = _ms(t0)

        relevancia = self._relevance(elegidos)
        umbral = self._threshold(elegidos)
        reescrita = consulta if consulta != question else None

        def metricas(generation_ms: int, answered: bool) -> AnswerMetrics:
            return AnswerMetrics(
                latency_ms=_ms(inicio),
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                generation_ms=generation_ms,
                top_score=relevancia,
                answered=answered,
                rewritten_query=reescrita,
            )

        # Guardrail: sin contexto suficientemente relevante no se consulta al modelo
        if not elegidos or relevancia is None or relevancia < umbral:
            logger.info(
                "Pregunta sin contexto relevante",
                extra={"top_score": relevancia, "threshold": umbral},
            )
            yield StreamEvent(token=prompts.NO_INFO)
            yield StreamEvent(
                final=RagAnswer(prompts.NO_INFO, (), metricas(0, answered=False), tuple(elegidos))
            )
            return

        mensajes = prompts.build_answer_messages(question, history, elegidos)
        t0 = time.perf_counter()
        partes: list[str] = []
        try:
            for fragmento in self._llm.stream(mensajes):
                partes.append(fragmento)
                yield StreamEvent(token=fragmento)
        except LLMUnavailableError as exc:
            logger.error("El modelo no está disponible", extra={"error": str(exc)})
            relleno = prompts.RESPUESTA_CORTADA if partes else prompts.LLM_CAIDO
            partes.append(relleno)
            yield StreamEvent(token=relleno)
            texto = "".join(partes)
            yield StreamEvent(
                final=RagAnswer(
                    texto,
                    self._citations(texto, elegidos),
                    metricas(_ms(t0), False),
                    tuple(elegidos),
                )
            )
            return
        except LLMResponseError as exc:
            logger.error("Respuesta inválida del modelo", extra={"error": str(exc)})
            raise

        texto = "".join(partes).strip()
        if not texto:
            raise LLMResponseError("El modelo devolvió una respuesta vacía")
        yield StreamEvent(
            final=RagAnswer(
                texto, self._citations(texto, elegidos), metricas(_ms(t0), True), tuple(elegidos)
            )
        )

    # --- Pasos --------------------------------------------------------------------------------
    def _rewrite(self, question: str, history: Sequence[Message]) -> str:
        """Reescribe preguntas de seguimiento; ante cualquier fallo usa la pregunta original."""
        if not history or not self._settings.query_rewriting_enabled:
            return question
        try:
            reescrita = self._llm.generate(
                prompts.build_rewrite_messages(question, history), max_tokens=120, temperature=0.0
            )
        except RagError as exc:
            logger.warning("No se pudo reescribir la pregunta", extra={"error": str(exc)})
            return question
        reescrita = reescrita.strip().strip('"').splitlines()[0].strip() if reescrita else ""
        if not reescrita or len(reescrita) > max(300, 3 * len(question)):
            return question
        return reescrita

    def _rerank(self, query: str, candidatos: list[RetrievedChunk]) -> list[RetrievedChunk]:
        s = self._settings
        try:
            return self._reranker.rerank(query, candidatos[: s.rerank_candidates], s.rerank_top_n)
        except RerankerError as exc:
            # Degradación: se conserva el orden de la recuperación
            logger.warning(
                "Reranker no disponible; se usa el orden original", extra={"error": str(exc)}
            )
            return candidatos[: s.rerank_top_n]

    @staticmethod
    def _relevance(elegidos: Sequence[RetrievedChunk]) -> float | None:
        if not elegidos:
            return None
        mejor = elegidos[0]
        if mejor.source == RetrievalSource.RERANKED:
            return mejor.score
        return max((r.similarity or 0.0) for r in elegidos)

    def _threshold(self, elegidos: Sequence[RetrievedChunk]) -> float:
        reranqueado = bool(elegidos) and elegidos[0].source == RetrievalSource.RERANKED
        s = self._settings
        return s.min_relevance_score if reranqueado else s.min_vector_similarity

    @staticmethod
    def _citations(texto: str, elegidos: Sequence[RetrievedChunk]) -> tuple[Citation, ...]:
        """Fuentes realmente citadas con [n]; si el modelo no citó, las más relevantes."""
        citados = {int(n) for n in _CITA.findall(texto) if 1 <= int(n) <= len(elegidos)}
        sin_cita = range(1, min(_MAX_FUENTES_SIN_CITA, len(elegidos)) + 1)
        indices = sorted(citados) if citados else list(sin_cita)
        vistos: set[str] = set()
        citas = []
        for i in indices:
            c = elegidos[i - 1].chunk
            if c.url in vistos:
                continue
            vistos.add(c.url)
            citas.append(Citation(c.title, c.url, c.bank))
        return tuple(citas)
