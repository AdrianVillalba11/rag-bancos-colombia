import pytest

from rag.conversation.memory import ConversationMemory, new_session_id, validate_session_id
from rag.domain.errors import InputValidationError
from rag.domain.models import AnswerMetrics, Message, Role


class RepoFalso:
    def __init__(self):
        self.mensajes: dict[str, list[Message]] = {}
        self.limites: list[int] = []

    def get_recent(self, session_id, limit):
        self.limites.append(limit)
        return self.mensajes.get(session_id, [])[-limit:] if limit > 0 else []

    def add_user_message(self, session_id, content, bank_filter):
        self.mensajes.setdefault(session_id, []).append(Message(Role.USER, content))
        return len(self.mensajes[session_id])

    def add_assistant_message(self, session_id, content, metrics, citations):
        self.mensajes.setdefault(session_id, []).append(Message(Role.ASSISTANT, content))
        return len(self.mensajes[session_id])


SESION = "sesion-de-prueba-1"


def test_los_id_generados_son_validos():
    assert validate_session_id(new_session_id())


IDS_MALOS = ["", "corto", "con espacios en el id", "a" * 65, "id;DROP TABLE x"]


@pytest.mark.parametrize("malo", IDS_MALOS)
def test_rechaza_ids_de_sesion_invalidos(malo):
    with pytest.raises(InputValidationError):
        validate_session_id(malo)


def test_la_ventana_devuelve_solo_los_ultimos_n_mensajes():
    repo = RepoFalso()
    memoria = ConversationMemory(repo, max_messages=4)
    for i in range(5):
        memoria.record_question(SESION, f"pregunta {i}", None)
        memoria.record_answer(SESION, f"respuesta {i}", AnswerMetrics(latency_ms=1), ())
    historial = memoria.history(SESION)
    assert [m.content for m in historial] == [
        "pregunta 3", "respuesta 3", "pregunta 4", "respuesta 4"
    ]  # fmt: skip
    assert repo.limites == [4]


def test_con_n_cero_no_hay_memoria():
    memoria = ConversationMemory(RepoFalso(), max_messages=0)
    memoria.record_question(SESION, "hola", None)
    assert memoria.history(SESION) == []


def test_las_sesiones_no_se_mezclan():
    memoria = ConversationMemory(RepoFalso(), max_messages=6)
    memoria.record_question("sesion-uno-aaaa", "de la uno", None)
    memoria.record_question("sesion-dos-bbbb", "de la dos", None)
    assert [m.content for m in memoria.history("sesion-uno-aaaa")] == ["de la uno"]


def test_la_memoria_valida_el_id_antes_de_consultar():
    repo = RepoFalso()
    with pytest.raises(InputValidationError):
        ConversationMemory(repo, 6).history("mal id")
    assert repo.limites == []
