import json
import threading

import pytest
from fastapi.testclient import TestClient

from rag.api.dependencies import AppState
from rag.api.main import create_app
from rag.config import Settings
from rag.conversation.memory import ConversationMemory
from rag.domain.errors import LLMResponseError
from rag.domain.models import (
    Role,
)
from rag.generation import prompts
from tests.unit.fakes import LLMFalso, RepoFalso, ServicioFalso, StoreFalso

SESION = "sesion-de-prueba-abc"


@pytest.fixture
def entorno():
    settings = Settings(max_question_length=200, history_max_messages=4)
    repo, servicio = RepoFalso(), ServicioFalso()
    estado = AppState(
        settings=settings, service=servicio, memory=ConversationMemory(repo, 4),
        repository=repo, store=StoreFalso(), llm=LLMFalso(), warm=threading.Event(),
    )  # fmt: skip
    with TestClient(create_app(estado)) as cliente:
        yield cliente, repo, servicio, estado


def eventos(respuesta):
    salida = []
    for bloque in respuesta.text.strip().split("\n\n"):
        nombre, datos = bloque.split("\n", 1)
        salida.append((nombre.removeprefix("event: "), json.loads(datos.removeprefix("data: "))))
    return salida


# --- Chat -------------------------------------------------------------------------------------
def test_el_chat_emite_meta_tokens_y_el_resultado_final(entorno):
    cliente, repo, _, _ = entorno
    r = cliente.post("/api/chat", json={"message": "¿Qué requisitos hay?", "session_id": SESION})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = eventos(r)
    assert [e[0] for e in evs] == ["meta", "token", "token", "done"]
    assert evs[0][1]["session_id"] == SESION
    assert "".join(d["text"] for n, d in evs if n == "token") == "Necesitas cédula [1]."
    hecho = evs[-1][1]
    assert hecho["answered"] and hecho["citations"][0]["bank"] == "davivienda"
    assert [f["role"] for f in repo.filas] == [Role.USER, Role.ASSISTANT]  # persiste ambos


def test_sin_session_id_se_crea_uno_y_se_devuelve(entorno):
    cliente, *_ = entorno
    evs = eventos(cliente.post("/api/chat", json={"message": "hola"}))
    assert len(evs[0][1]["session_id"]) >= 8


def test_el_historial_se_pasa_al_servicio_sin_incluir_la_pregunta_actual(entorno):
    cliente, _, servicio, _ = entorno
    cliente.post("/api/chat", json={"message": "primera", "session_id": SESION})
    cliente.post("/api/chat", json={"message": "segunda", "session_id": SESION, "bank": "BBVA"})
    pregunta, historial, banco = servicio.llamadas[1]
    assert pregunta == "segunda" and banco == "bbva"
    assert [m.content for m in historial] == ["primera", "Necesitas cédula [1]."]


def test_modo_sin_streaming_devuelve_json(entorno):
    cliente, *_ = entorno
    r = cliente.post("/api/chat", json={"message": "hola", "session_id": SESION, "stream": False})
    cuerpo = r.json()
    assert r.status_code == 200 and cuerpo["answer"].startswith("Necesitas")
    assert cuerpo["session_id"] == SESION and cuerpo["message_id"] == 2


@pytest.mark.parametrize(
    ("cuerpo", "codigo"),
    [
        ({"message": "   "}, "invalid_input"),
        ({"message": "x" * 201}, "invalid_input"),
        ({"message": "hola", "session_id": "id malo!"}, "invalid_input"),
        ({"message": "hola", "bank": "inventado"}, "unknown_bank"),
    ],
)
def test_entradas_invalidas_devuelven_422_con_mensaje_claro(entorno, cuerpo, codigo):
    cliente, repo, *_ = entorno
    r = cliente.post("/api/chat", json=cuerpo)
    assert r.status_code == 422 and r.json()["error"]["code"] == codigo
    assert repo.filas == []  # no se guarda nada


def test_cuerpo_malformado_devuelve_422_generico(entorno):
    cliente, *_ = entorno
    r = cliente.post("/api/chat", json={"sin_mensaje": True})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"


def test_los_caracteres_de_control_se_eliminan_de_la_pregunta(entorno):
    cliente, _, servicio, _ = entorno
    cliente.post("/api/chat", json={"message": "ho\x00la\x07"})
    assert servicio.llamadas[0][0] == "hola"


def test_un_error_durante_el_streaming_viaja_como_evento_sin_detalles_internos(entorno):
    cliente, _, servicio, _ = entorno
    servicio.falla = LLMResponseError("traza interna secreta")
    evs = eventos(cliente.post("/api/chat", json={"message": "hola"}))
    nombre, datos = evs[-1]
    assert nombre == "error" and "secreta" not in json.dumps(datos)


# --- Historial, feedback y configuración ------------------------------------------------------
def test_devuelve_el_historial_de_una_sesion(entorno):
    cliente, *_ = entorno
    cliente.post("/api/chat", json={"message": "hola", "session_id": SESION})
    mensajes = cliente.get(f"/api/sessions/{SESION}/messages").json()
    assert [m["role"] for m in mensajes] == ["user", "assistant"]
    assert mensajes[1]["citations"][0]["url"] == "https://x.com/a"


def test_lista_solo_las_conversaciones_cuyos_id_envia_el_cliente(entorno):
    cliente, *_ = entorno
    cliente.post("/api/chat", json={"message": "pregunta de A", "session_id": "sesion-aaaa-1111"})
    cliente.post("/api/chat", json={"message": "pregunta de B", "session_id": "sesion-bbbb-2222"})
    r = cliente.get("/api/sessions", params={"ids": "sesion-aaaa-1111"})
    assert [s["title"] for s in r.json()] == ["pregunta de A"]
    assert r.json()[0]["messages"] == 2


def test_el_listado_ignora_ids_invalidos_o_inexistentes_y_sin_ids_no_lista_nada(entorno):
    cliente, *_ = entorno
    cliente.post("/api/chat", json={"message": "hola", "session_id": "sesion-aaaa-1111"})
    ids = "mal id,';DROP TABLE x,sesion-inexistente-9,sesion-aaaa-1111"
    assert len(cliente.get("/api/sessions", params={"ids": ids}).json()) == 1
    assert cliente.get("/api/sessions").json() == []  # no hay listado global


def test_el_historial_valida_el_id(entorno):
    cliente, *_ = entorno
    assert cliente.get("/api/sessions/mal id/messages").status_code in (404, 422)


def test_feedback_valido_invalido_e_inexistente(entorno):
    cliente, repo, *_ = entorno
    cliente.post("/api/chat", json={"message": "hola", "session_id": SESION})
    assert cliente.post("/api/messages/2/feedback", json={"value": 1}).status_code == 200
    assert repo.feedback == {2: 1}
    assert cliente.post("/api/messages/2/feedback", json={"value": 7}).status_code == 422
    r = cliente.post("/api/messages/99/feedback", json={"value": 1})
    assert r.status_code == 404 and r.json()["error"]["code"] == "session_not_found"


def test_configuracion_para_la_interfaz(entorno):
    cliente, *_ = entorno
    cfg = cliente.get("/api/config").json()
    assert {b["id"] for b in cfg["banks"]} == {"bbva", "bancolombia", "davivienda"}
    assert cfg["max_question_length"] == 200 and cfg["history_max_messages"] == 4


# --- Salud, interfaz y seguridad --------------------------------------------------------------
def test_health_ok_y_degradado(entorno):
    cliente, repo, *_ = entorno
    r = cliente.get("/api/health")
    assert r.status_code == 200 and r.json()["indexed_chunks"] == 3235
    repo.vivo = False
    r = cliente.get("/api/health")
    assert r.status_code == 503 and r.json()["services"]["postgres"] is False


def test_sirve_la_interfaz_y_sus_recursos_con_cabeceras_de_seguridad(entorno):
    cliente, *_ = entorno
    pagina = cliente.get("/")
    assert pagina.status_code == 200 and "Asistente de bancos colombianos" in pagina.text
    assert "script-src 'self'" in pagina.headers["content-security-policy"]
    assert pagina.headers["x-content-type-options"] == "nosniff"
    assert cliente.get("/static/app.js").status_code == 200
    assert cliente.get("/static/styles.css").status_code == 200


def test_la_interfaz_no_usa_innerhtml_ni_scripts_en_linea(entorno):
    cliente, *_ = entorno
    assert ".innerHTML" not in cliente.get("/static/app.js").text
    html = cliente.get("/").text
    assert "<script>" not in html and " onclick=" not in html and " style=" not in html


def test_los_errores_no_controlados_no_filtran_detalles(entorno):
    cliente, repo, *_ = entorno
    repo.add_user_message = lambda *a: (_ for _ in ()).throw(RuntimeError("clave=secreta"))
    with TestClient(cliente.app, raise_server_exceptions=False) as c2:
        r = c2.post("/api/chat", json={"message": "hola"})
    assert r.status_code == 500 and "secreta" not in r.text
    assert r.json()["error"]["code"] == "internal_error"


def test_el_mensaje_de_no_informacion_es_el_del_dominio():
    assert "No encontré" in prompts.NO_INFO


# --- Analítica --------------------------------------------------------------------------------
def test_analitica_calcula_el_informe_sobre_el_historico(entorno):
    cliente, *_ = entorno
    cliente.post(
        "/api/chat", json={"message": "¿Qué es una cuenta de ahorros?", "session_id": SESION}
    )
    r = cliente.get("/api/analytics")
    assert r.status_code == 200
    informe = r.json()
    for bloque in ("overview", "volume", "topics", "latency", "knowledge_gaps", "sources",
                   "feedback", "conversations"):  # fmt: skip
        assert bloque in informe
    assert informe["overview"]["questions"] == 1 and informe["overview"]["conversations"] == 1


def test_analitica_valida_el_parametro_de_dias(entorno):
    cliente, *_ = entorno
    assert cliente.get("/api/analytics", params={"days": 7}).status_code == 200
    assert cliente.get("/api/analytics", params={"days": 0}).status_code == 422


def test_analitica_exige_token_cuando_esta_configurado(entorno):
    cliente, _, _, estado = entorno
    estado.settings.analytics_token = "secreto-123"
    assert cliente.get("/api/analytics/protected").json() == {"protected": True}
    sin_token = cliente.get("/api/analytics")
    assert sin_token.status_code == 401 and sin_token.json()["error"]["code"] == "unauthorized"
    assert cliente.get("/api/analytics", headers={"X-Analytics-Token": "mal"}).status_code == 401
    ok = cliente.get("/api/analytics", headers={"X-Analytics-Token": "secreto-123"})
    assert ok.status_code == 200


def test_el_dashboard_se_sirve_y_no_usa_innerhtml(entorno):
    cliente, *_ = entorno
    pagina = cliente.get("/analytics")
    assert pagina.status_code == 200 and "Analítica del asistente" in pagina.text
    assert ".innerHTML" not in cliente.get("/static/analytics.js").text
    assert cliente.get("/static/analytics.css").status_code == 200
