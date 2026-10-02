import time

import pytest

from rag.domain.errors import OperationTimeoutError
from rag.infra.timeouts import call_with_timeout


def test_devuelve_el_resultado_si_termina_a_tiempo():
    assert call_with_timeout(lambda: 42, seconds=1) == 42


def test_lanza_error_de_dominio_al_superar_el_limite():
    with pytest.raises(OperationTimeoutError):
        call_with_timeout(lambda: time.sleep(0.5), seconds=0.05, operation="prueba")
