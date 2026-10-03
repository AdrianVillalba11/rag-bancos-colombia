"""Pruebas contra un PostgreSQL real. Se ejecutan dentro de Docker (docker compose run --rm tests)
y se omiten si la base no está disponible."""

import uuid

import pytest

from rag.config import Settings
from rag.conversation.repository import PostgresConversationRepository
from rag.domain.errors import ConversationRepositoryError, SessionNotFoundError
from rag.domain.models import AnswerMetrics, Citation, Role
from rag.infra.db import create_pool


@pytest.fixture(scope="module")
def repo():
    try:
        repositorio = PostgresConversationRepository(
            create_pool(Settings()), max_retries=0, open_timeout=3
        )
        repositorio.open()
    except Exception:
        pytest.skip("PostgreSQL no disponible")
    yield repositorio
    repositorio.close()


@pytest.fixture
def sesion(repo):
    sid = f"test-{uuid.uuid4().hex[:12]}"
    yield sid
    with repo._pool.connection() as conn:
        conn.execute("DELETE FROM conversations WHERE session_id = %s", (sid,))


METRICAS = AnswerMetrics(
    latency_ms=1200, retrieval_ms=100, rerank_ms=50, generation_ms=1000,
    top_score=0.93, answered=True, rewritten_query="pregunta reescrita",
)  # fmt: skip
CITAS = (
    Citation("Cuenta de ahorros", "https://x.com/a", "bbva"),
    Citation("T2", "https://x.com/b", "davivienda"),
)


def test_guarda_y_recupera_una_conversacion_con_metricas_y_citas(repo, sesion):
    repo.add_user_message(sesion, "¿Qué necesito?", "bbva")
    respuesta_id = repo.add_assistant_message(sesion, "Cédula [1]", METRICAS, CITAS)

    mensajes = repo.get_session(sesion)
    assert [m.role for m in mensajes] == [Role.USER, Role.ASSISTANT]
    usuario, asistente = mensajes
    assert usuario.bank_filter == "bbva"
    assert asistente.id == respuesta_id and asistente.latency_ms == 1200
    assert asistente.top_score == pytest.approx(0.93) and asistente.answered is True
    assert asistente.rewritten_query == "pregunta reescrita"
    assert [c.url for c in asistente.citations] == ["https://x.com/a", "https://x.com/b"]


def test_get_recent_devuelve_los_ultimos_n_en_orden_cronologico(repo, sesion):
    for i in range(4):
        repo.add_user_message(sesion, f"p{i}", None)
        repo.add_assistant_message(sesion, f"r{i}", METRICAS, ())
    recientes = repo.get_recent(sesion, 3)
    assert [m.content for m in recientes] == ["r2", "p3", "r3"]
    assert repo.get_recent(sesion, 0) == []
    assert repo.get_recent("sesion-que-no-existe", 5) == []


def test_las_sesiones_estan_aisladas(repo, sesion):
    otra = f"test-{uuid.uuid4().hex[:12]}"
    try:
        repo.add_user_message(sesion, "mensaje A", None)
        repo.add_user_message(otra, "mensaje B", None)
        assert [m.content for m in repo.get_recent(sesion, 10)] == ["mensaje A"]
    finally:
        with repo._pool.connection() as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = %s", (otra,))


def test_feedback_solo_en_respuestas_y_valores_validos(repo, sesion):
    pregunta_id = repo.add_user_message(sesion, "hola", None)
    respuesta_id = repo.add_assistant_message(sesion, "ok", METRICAS, ())
    repo.set_feedback(respuesta_id, 1)
    assert repo.get_session(sesion)[1].feedback == 1
    with pytest.raises(SessionNotFoundError):
        repo.set_feedback(pregunta_id, 1)
    with pytest.raises(ConversationRepositoryError):
        repo.set_feedback(respuesta_id, 5)


def test_iter_messages_recorre_el_historico_y_list_sessions_lo_refleja(repo, sesion):
    repo.add_user_message(sesion, "pregunta", None)
    repo.add_assistant_message(sesion, "respuesta", METRICAS, CITAS)
    propios = [m for m in repo.iter_messages() if m.session_id == sesion]
    assert [m.content for m in propios] == ["pregunta", "respuesta"]
    assert len(propios[1].citations) == 2
    assert sesion in repo.list_sessions(limit=500)
    assert repo.ping() is True


def test_el_esquema_es_idempotente(repo):
    repo.open()  # reabrir no falla ni duplica nada
    assert repo.ping()
