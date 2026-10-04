import threading
import time

import pytest
from fastapi.testclient import TestClient

from rag.api.dependencies import AppState
from rag.api.main import create_app
from rag.config import Settings
from rag.conversation.memory import ConversationMemory
from rag.domain.errors import VectorStoreError
from rag.infra.ratelimit import SlidingWindowRateLimiter
from rag.retrieval.vector_store import ChromaVectorStore
from tests.unit.fakes import LLMFalso, RepoFalso, ServicioFalso, StoreFalso

# --- Limitador ---------------------------------------------------------------------------------


class Reloj:
    def __init__(self):
        self.ahora = 1000.0

    def __call__(self):
        return self.ahora


def test_permite_hasta_el_limite_y_luego_indica_cuanto_esperar():
    reloj = Reloj()
    limitador = SlidingWindowRateLimiter(3, 60, clock=reloj)
    assert [limitador.hit("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    reloj.ahora += 10
    espera = limitador.hit("a")
    assert espera == pytest.approx(50)  # la primera marca vence a los 60 s


def test_la_ventana_se_desliza_y_vuelve_a_permitir():
    reloj = Reloj()
    limitador = SlidingWindowRateLimiter(2, 60, clock=reloj)
    limitador.hit("a")
    reloj.ahora += 30
    limitador.hit("a")
    assert limitador.hit("a") > 0
    reloj.ahora += 31  # vence la primera, queda una
    assert limitador.hit("a") == 0.0


def test_los_clientes_tienen_contadores_independientes():
    limitador = SlidingWindowRateLimiter(1, 60, clock=Reloj())
    assert limitador.hit("a") == 0.0
    assert limitador.hit("a") > 0
    assert limitador.hit("b") == 0.0


def test_las_peticiones_rechazadas_no_consumen_cupo():
    reloj = Reloj()
    limitador = SlidingWindowRateLimiter(1, 60, clock=reloj)
    limitador.hit("a")
    for _ in range(5):
        limitador.hit("a")
    reloj.ahora += 60
    assert limitador.hit("a") == 0.0  # los intentos rechazados no extendieron el bloqueo


def test_la_memoria_esta_acotada():
    reloj = Reloj()
    limitador = SlidingWindowRateLimiter(5, 60, clock=reloj, max_keys=50)
    for i in range(500):
        reloj.ahora += 0.01
        limitador.hit(f"ip-{i}")
    assert len(limitador._hits) <= 51


def test_es_seguro_con_hilos():
    limitador = SlidingWindowRateLimiter(100, 60)
    permitidas = []

    def trabajo():
        permitidas.extend(limitador.hit("a") == 0.0 for _ in range(50))

    hilos = [threading.Thread(target=trabajo) for _ in range(8)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    assert sum(permitidas) == 100  # 400 intentos, exactamente 100 permitidos


def test_valida_los_parametros():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(0)


# --- API ---------------------------------------------------------------------------------------


def construir(**cfg):
    settings = Settings(**cfg)
    repo = RepoFalso()
    estado = AppState(
        settings=settings,
        service=ServicioFalso(),
        memory=ConversationMemory(repo, 4),
        repository=repo,
        store=StoreFalso(),
        llm=LLMFalso(),
        warm=threading.Event(),
    )
    return estado


def cliente(estado):
    return TestClient(create_app(estado))


def test_el_chat_responde_429_con_retry_after_al_superar_el_limite():
    estado = construir(rate_limit_per_minute=3)
    with cliente(estado) as c:
        for _ in range(3):
            assert c.post("/api/chat", json={"message": "hola"}).status_code == 200
        r = c.post("/api/chat", json={"message": "hola"})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(r.headers["retry-after"]) >= 1
    assert "Inténtalo de nuevo" in r.json()["error"]["message"]


def test_el_limite_tambien_protege_el_feedback_y_no_el_resto():
    estado = construir(rate_limit_per_minute=1)
    with cliente(estado) as c:
        c.post("/api/chat", json={"message": "hola", "session_id": "sesion-limite-01"})
        assert c.post("/api/messages/2/feedback", json={"value": 1}).status_code == 429
        assert c.get("/api/health").status_code == 200  # la salud no se limita
        assert c.get("/api/config").status_code == 200


def test_clientes_distintos_no_comparten_limite_cuando_se_confia_en_el_proxy():
    estado = construir(rate_limit_per_minute=1, trust_proxy_headers=True)
    with cliente(estado) as c:
        a = {"X-Forwarded-For": "10.0.0.1, 172.16.0.1"}
        b = {"X-Forwarded-For": "10.0.0.2"}
        assert c.post("/api/chat", json={"message": "hola"}, headers=a).status_code == 200
        assert c.post("/api/chat", json={"message": "hola"}, headers=a).status_code == 429
        assert c.post("/api/chat", json={"message": "hola"}, headers=b).status_code == 200


def test_sin_confiar_en_el_proxy_la_cabecera_falsificada_no_evade_el_limite():
    estado = construir(rate_limit_per_minute=1, trust_proxy_headers=False)
    with cliente(estado) as c:
        assert (
            c.post(
                "/api/chat", json={"message": "hola"}, headers={"X-Forwarded-For": "1.1.1.1"}
            ).status_code
            == 200
        )
        r = c.post("/api/chat", json={"message": "hola"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert r.status_code == 429


def test_responde_503_cuando_no_quedan_cupos_de_concurrencia():
    estado = construir(max_concurrent_chats=1)
    estado.chat_slots.acquire()  # otro usuario ocupa el único cupo
    with cliente(estado) as c:
        r = c.post("/api/chat", json={"message": "hola"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "service_busy"
    assert int(r.headers["retry-after"]) >= 1


def test_el_cupo_se_libera_tras_cada_respuesta_streaming_y_json():
    estado = construir(max_concurrent_chats=1)
    with cliente(estado) as c:
        for stream in (True, False, True):
            r = c.post("/api/chat", json={"message": "hola", "stream": stream})
            assert r.status_code == 200
    assert estado.chat_slots.acquire(blocking=False)  # el cupo sigue disponible


def test_el_cupo_se_libera_si_falla_el_servicio_o_la_base():
    estado = construir(max_concurrent_chats=1)
    estado.repository.add_user_message = lambda *a: (_ for _ in ()).throw(RuntimeError("bd"))
    with TestClient(create_app(estado), raise_server_exceptions=False) as c:
        assert c.post("/api/chat", json={"message": "hola"}).status_code == 500
    assert estado.chat_slots.acquire(blocking=False)


def test_rechaza_cuerpos_demasiado_grandes_antes_de_procesarlos():
    estado = construir(max_request_bytes=2048)
    with cliente(estado) as c:
        r = c.post("/api/chat", json={"message": "x" * 5000})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    assert estado.repository.filas == []


# --- Timeout de ChromaDB -----------------------------------------------------------------------
class ColeccionColgada:
    def count(self):
        time.sleep(2)


class ClienteColgado:
    def get_or_create_collection(self, **_):
        return ColeccionColgada()


def test_una_llamada_colgada_a_chroma_termina_con_error_de_dominio():
    store = ChromaVectorStore("x", 1, "c", client=ClienteColgado(), max_retries=0, timeout=0.2)
    inicio = time.perf_counter()
    with pytest.raises(VectorStoreError):
        store.count()
    assert time.perf_counter() - inicio < 1.5
