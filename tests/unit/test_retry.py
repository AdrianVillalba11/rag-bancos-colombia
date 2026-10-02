import pytest

from rag.infra.retry import compute_delay, retry


def test_reintenta_hasta_tener_exito():
    llamadas = []
    esperas = []

    @retry(max_retries=3, base_delay=1, jitter=False, sleep=esperas.append)
    def inestable():
        llamadas.append(1)
        if len(llamadas) < 3:
            raise ConnectionError("falla")
        return "ok"

    assert inestable() == "ok"
    assert len(llamadas) == 3
    assert esperas == [1, 2]  # backoff exponencial


def test_propaga_el_error_al_agotar_reintentos():
    llamadas = []

    @retry(max_retries=2, jitter=False, sleep=lambda _: None)
    def siempre_falla():
        llamadas.append(1)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        siempre_falla()
    assert len(llamadas) == 3  # 1 intento + 2 reintentos


def test_no_reintenta_excepciones_fuera_de_retry_on():
    llamadas = []

    @retry(max_retries=5, retry_on=(ConnectionError,), sleep=lambda _: None)
    def otro_error():
        llamadas.append(1)
        raise KeyError("x")

    with pytest.raises(KeyError):
        otro_error()
    assert len(llamadas) == 1


def test_should_retry_puede_descartar_el_reintento():
    llamadas = []

    @retry(max_retries=5, should_retry=lambda exc: "temporal" in str(exc), sleep=lambda _: None)
    def permanente():
        llamadas.append(1)
        raise RuntimeError("definitivo")

    with pytest.raises(RuntimeError):
        permanente()
    assert len(llamadas) == 1


def test_sin_reintentos_ejecuta_una_sola_vez():
    llamadas = []

    @retry(max_retries=0, sleep=lambda _: None)
    def falla():
        llamadas.append(1)
        raise OSError

    with pytest.raises(OSError):
        falla()
    assert len(llamadas) == 1


def test_el_delay_respeta_el_maximo():
    assert compute_delay(10, base_delay=1, max_delay=5, jitter=False) == 5
    assert 0 <= compute_delay(3, base_delay=1, max_delay=5, jitter=True) <= 4
