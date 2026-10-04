"""Los perfiles de configuración (CPU y GPU) solo deben diferir en lo propio de cada uno."""

from pathlib import Path

import pytest

from rag.config import Settings

RAIZ = Path(__file__).resolve().parents[2]
CPU = RAIZ / ".env.example"
GPU = RAIZ / ".env.gpu.example"

#: Claves que solo existen en el perfil GPU (activan la GPU en Docker Compose)
SOLO_GPU = {"COMPOSE_PATH_SEPARATOR", "COMPOSE_FILE", "TORCH_INDEX"}
#: Claves cuyo valor es distinto a propósito entre perfiles
DIFERENTES = {
    "LLM_MODEL",
    "LLM_NUM_CTX",
    "LLM_MAX_ANSWER_TOKENS",
    "RERANKER_DEVICE",
    "RERANK_CANDIDATES",
}
#: Variables que lee Docker Compose y no la aplicación
SOLO_COMPOSE = {"APP_PORT"} | SOLO_GPU


def leer(ruta: Path) -> dict[str, str]:
    valores = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            valores[clave.strip()] = valor.strip()
    return valores


def test_ambos_perfiles_existen_y_se_documentan_mutuamente():
    assert "cp .env.gpu.example .env" in CPU.read_text(encoding="utf-8")
    assert "cp .env.example .env" in GPU.read_text(encoding="utf-8")


def test_los_perfiles_tienen_las_mismas_claves_salvo_las_de_la_gpu():
    assert set(leer(GPU)) - set(leer(CPU)) == SOLO_GPU
    assert set(leer(CPU)) - set(leer(GPU)) == set()


def test_los_valores_solo_difieren_en_las_claves_propias_de_cada_perfil():
    cpu, gpu = leer(CPU), leer(GPU)
    distintas = {k for k in cpu if cpu[k] != gpu[k]}
    assert distintas == DIFERENTES


def test_el_perfil_cpu_usa_el_modelo_ligero_y_el_gpu_el_completo():
    cpu, gpu = leer(CPU), leer(GPU)
    assert cpu["LLM_MODEL"] == "llama3.2:3b" and cpu["RERANKER_DEVICE"] == "cpu"
    assert gpu["LLM_MODEL"] == "llama3" and gpu["RERANKER_DEVICE"] == "cuda"
    assert "cu" in gpu["TORCH_INDEX"] and "cpu" not in gpu["TORCH_INDEX"]
    assert "docker-compose.gpu.yml" in gpu["COMPOSE_FILE"]


@pytest.mark.parametrize("perfil", [CPU, GPU], ids=["cpu", "gpu"])
def test_cada_perfil_es_una_configuracion_valida(perfil):
    # Se valida sin tocar el entorno ni un .env local
    Settings(_env_file=perfil)  # type: ignore[call-arg]


@pytest.mark.parametrize("perfil", [CPU, GPU], ids=["cpu", "gpu"])
def test_todas_las_claves_corresponden_a_un_parametro_real(perfil):
    campos = {nombre.upper() for nombre in Settings.model_fields}
    desconocidas = set(leer(perfil)) - campos - SOLO_COMPOSE
    assert desconocidas == set()
