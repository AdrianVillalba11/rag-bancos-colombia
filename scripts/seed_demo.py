"""Genera conversaciones de demostración con el sistema real, para probar la analítica.

Cada pregunta pasa por el pipeline completo (recuperación, reranker, modelo) y se guarda en
PostgreSQL igual que las de un usuario. Las sesiones llevan el prefijo `demo-` para poder
identificarlas. NO se fabrican valoraciones (👍/👎): esas solo las ponen los usuarios.

Uso (dentro de Docker):
    docker compose run --rm ingest python scripts/seed_demo.py
"""

import sys
import time

from rag.api.dependencies import build_conversation_repository, build_memory, build_rag_service
from rag.config import get_settings
from rag.infra.logging import configure_logging

# (ID de sesión, banco, [preguntas]); las preguntas de seguimiento dependen de la anterior
CONVERSACIONES: list[tuple[str, str | None, list[str]]] = [
    (
        "demo-cuentas-davivienda",
        "davivienda",
        [
            "¿Qué requisitos necesito para abrir una cuenta de ahorros?",
            "¿Y cuánto cuesta la cuota de manejo?",
            "¿Qué otros tipos de cuenta tienen?",
        ],
    ),
    (
        "demo-tarjetas-bancolombia",
        "bancolombia",
        [
            "¿Qué tarjetas de crédito tienen cuota de manejo de $0?",
            "¿Y cuáles son las tasas de interés de esas tarjetas?",
        ],
    ),
    (
        "demo-credito-bbva",
        "bbva",
        [
            "¿Cómo solicito un crédito de libre inversión?",
            "¿Qué documentos debo presentar?",
            "¿Puedo pagarlo anticipadamente sin costo?",
        ],
    ),
    (
        "demo-4x1000",
        None,
        [
            "¿Qué es el 4x1000 y cómo se exonera?",
        ],
    ),
    (
        "demo-cdt-davivienda",
        "davivienda",
        [
            "¿Qué es un CDT y cómo funciona?",
            "¿Cuál es el monto mínimo para abrirlo?",
        ],
    ),
    (
        "demo-seguros",
        None,
        [
            "¿Qué seguros ofrecen para el hogar y el vehículo?",
        ],
    ),
    (
        "demo-app-bancolombia",
        "bancolombia",
        [
            "¿Cómo activo la clave dinámica en la app?",
            "¿Qué hago si pierdo mi tarjeta débito?",
        ],
    ),
    (
        "demo-nomina",
        None,
        [
            "¿Qué beneficios tiene la cuenta de nómina?",
            "¿Qué beneficios tiene la cuenta de nómina?",  # repetida (preguntas frecuentes)
        ],
    ),
    (
        "demo-vivienda-bbva",
        "bbva",
        [
            "¿Qué requisitos tiene un crédito hipotecario para comprar vivienda?",
        ],
    ),
    (
        "demo-linea-atencion",
        None,
        [
            "¿Cuáles son las líneas de atención al cliente?",
        ],
    ),
    (
        "demo-fuera-de-tema-1",
        None,
        [
            "¿Quién ganó el mundial de fútbol de 2022?",
        ],
    ),
    (
        "demo-fuera-de-tema-2",
        None,
        [
            "¿Cuál es el precio del bitcoin hoy?",
        ],
    ),
    (
        "demo-cobertura-limitada",
        "davivienda",
        [
            "¿Qué tasa de interés exacta tiene hoy el crédito de vehículo?",
        ],
    ),
]


def main() -> int:
    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    repositorio = build_conversation_repository(settings)
    memoria = build_memory(repositorio, settings)
    servicio = build_rag_service(settings)
    servicio.warmup()

    total = sum(len(p) for _, _, p in CONVERSACIONES)
    hechas = 0
    try:
        for sesion, banco, preguntas in CONVERSACIONES:
            for pregunta in preguntas:
                historial = memoria.history(sesion)
                memoria.record_question(sesion, pregunta, banco)
                final = servicio.answer(pregunta, historial, banco)
                memoria.record_answer(sesion, final.text, final.metrics, final.citations)
                hechas += 1
                estado = "ok " if final.metrics.answered else "sin"
                ms = final.metrics.latency_ms
                print(f"[{hechas:>2}/{total}] {estado} {ms:>6} ms  {pregunta}")
                sys.stdout.flush()
                time.sleep(0.2)
    finally:
        repositorio.close()
    print(f"\n{hechas} preguntas en {len(CONVERSACIONES)} conversaciones de demostración.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
