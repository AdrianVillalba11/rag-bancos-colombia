"""Chat por línea de comandos con historial persistente en PostgreSQL.

Uso (dentro de Docker):
    docker compose run --rm ingest python scripts/chat.py                  # sesión nueva
    docker compose run --rm ingest python scripts/chat.py --session ID     # retoma una sesión
    docker compose run --rm ingest python scripts/chat.py --bank davivienda
"""

import argparse

from rag.api.dependencies import build_conversation_repository, build_memory, build_rag_service
from rag.config import get_settings
from rag.conversation.memory import new_session_id, validate_session_id
from rag.infra.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", help="Filtra las respuestas a un banco")
    parser.add_argument("--session", help="ID de una sesión existente para retomarla")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    repositorio = build_conversation_repository(settings)
    memoria = build_memory(repositorio, settings)
    servicio = build_rag_service(settings)

    sesion = validate_session_id(args.session) if args.session else new_session_id()
    previos = memoria.history(sesion)
    print(f"Sesión: {sesion} (recuerda los últimos {memoria.max_messages} mensajes; "
          f"{len(previos)} cargados). Escribe tu pregunta (vacío para salir).")

    while True:
        try:
            pregunta = input("\nTú: ").strip()
        except EOFError:
            break
        if not pregunta:
            break
        historial = memoria.history(sesion)  # antes de guardar la pregunta actual
        memoria.record_question(sesion, pregunta, args.bank)
        print("Asistente: ", end="", flush=True)
        final = None
        for evento in servicio.stream_answer(pregunta, historial, args.bank):
            if evento.token:
                print(evento.token, end="", flush=True)
            if evento.final:
                final = evento.final
        print()
        if final:
            memoria.record_answer(sesion, final.text, final.metrics, final.citations)
            for c in final.citations:
                print(f"  - [{c.bank}] {c.title}: {c.url}")
            m = final.metrics
            print(f"  ({m.latency_ms} ms; relevancia {m.top_score})")
    repositorio.close()


if __name__ == "__main__":
    main()
