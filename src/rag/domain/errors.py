"""Excepciones de dominio.

Todas heredan de `RagError`. Cada una lleva un `code` estable, pensado para que la capa de API
las traduzca a respuestas coherentes sin acoplar el dominio a HTTP.
"""


class RagError(Exception):
    """Error base de la aplicación."""

    code = "rag_error"
    #: Indica si reintentar la misma operación tiene sentido
    retryable = False

    def __init__(self, message: str = "", *, cause: BaseException | None = None) -> None:
        super().__init__(message or self.__class__.__name__)
        if cause is not None:
            self.__cause__ = cause


# --- Entradas del usuario ---------------------------------------------------------------------
class InputValidationError(RagError):
    code = "invalid_input"


class UnknownBankError(InputValidationError):
    code = "unknown_bank"


class RateLimitExceededError(RagError):
    code = "rate_limit_exceeded"


# --- Ingesta ----------------------------------------------------------------------------------
class ScrapingError(RagError):
    code = "scraping_error"
    retryable = True


class RobotsDisallowedError(ScrapingError):
    """La URL está prohibida por el robots.txt del sitio."""

    code = "robots_disallowed"
    retryable = False


class HttpStatusError(ScrapingError):
    """Respuesta HTTP no exitosa. Solo es reintentable si el fallo parece transitorio."""

    code = "http_status_error"

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"HTTP {status_code} al solicitar {url}")
        self.status_code = status_code
        self.url = url
        self.retryable = status_code == 429 or status_code >= 500


class ScrapingBlockedError(HttpStatusError):
    """El sitio rechaza el acceso automatizado (401/403). No se intenta evadir el bloqueo."""

    code = "scraping_blocked"

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(status_code, url)
        self.retryable = False


class CleaningError(RagError):
    code = "cleaning_error"


class IngestionError(RagError):
    code = "ingestion_error"


# --- Recuperación -----------------------------------------------------------------------------
class EmbeddingError(RagError):
    code = "embedding_error"
    retryable = True


class VectorStoreError(RagError):
    code = "vector_store_error"
    retryable = True


class RerankerError(RagError):
    code = "reranker_error"


class RetrievalError(RagError):
    code = "retrieval_error"


# --- Generación -------------------------------------------------------------------------------
class LLMUnavailableError(RagError):
    """El servicio del modelo no responde o agotó los reintentos."""

    code = "llm_unavailable"
    retryable = True


class LLMResponseError(RagError):
    """El modelo respondió, pero de forma inválida o vacía."""

    code = "llm_response_error"


# --- Conversación -----------------------------------------------------------------------------
class ConversationRepositoryError(RagError):
    code = "conversation_repository_error"
    retryable = True


class SessionNotFoundError(RagError):
    code = "session_not_found"


# --- Infraestructura --------------------------------------------------------------------------
class OperationTimeoutError(RagError):
    code = "operation_timeout"
    retryable = True
