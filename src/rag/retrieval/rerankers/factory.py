"""Factory de rerankers: elige la estrategia según la configuración."""

from __future__ import annotations

from rag.config import Settings
from rag.domain.interfaces import Reranker
from rag.retrieval.rerankers.base import NoopReranker
from rag.retrieval.rerankers.bge import BgeReranker


class RerankerFactory:
    @staticmethod
    def create(settings: Settings) -> Reranker:
        if not settings.reranker_enabled:
            return NoopReranker()
        return BgeReranker(
            settings.reranker_model,
            device=settings.reranker_device,
            max_length=settings.reranker_max_length,
        )
