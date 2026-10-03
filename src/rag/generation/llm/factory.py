"""Factory de clientes de lenguaje (hoy Ollama; el sistema depende solo de LLMClient)."""

from __future__ import annotations

from rag.config import Settings
from rag.domain.interfaces import LLMClient
from rag.generation.llm.ollama import OllamaLLM


class LLMFactory:
    @staticmethod
    def create(settings: Settings) -> LLMClient:
        return OllamaLLM(
            settings.ollama_base_url,
            settings.llm_model,
            num_ctx=settings.llm_num_ctx,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            keep_alive=settings.llm_keep_alive,
            max_tokens=settings.llm_max_answer_tokens,
        )
