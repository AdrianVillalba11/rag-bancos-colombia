"""Chat por línea de comandos sobre el servicio RAG (historial solo en memoria).

Uso (dentro de Docker):
    docker compose run --rm ingest python scripts/chat.py
    docker compose run --rm ingest python scripts/chat.py --bank davivienda
"""

import argparse

from rag.api.dependencies import build_rag_service
from rag.config import get_settings
from rag.domain.models import Message, Role
from rag.infra.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", help="Filtra las respuestas a un banco")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    service = build_rag_service(settings)
    historial: list[Message] = []

    print("Escribe tu pregunta (vacío para salir).")
    while True:
        try:
            pregunta = input("\nTú: ").strip()
        except EOFError:
            break
        if not pregunta:
            break
        print("Asistente: ", end="", flush=True)
        final = None
        for evento in service.stream_answer(pregunta, historial, args.bank):
            if evento.token:
                print(evento.token, end="", flush=True)
            if evento.final:
                final = evento.final
        print()
        if final:
            for c in final.citations:
                print(f"  - [{c.bank}] {c.title}: {c.url}")
            m = final.metrics
            print(f"  ({m.latency_ms} ms; recuperación {m.retrieval_ms}, rerank {m.rerank_ms}, "
                  f"generación {m.generation_ms}; relevancia {m.top_score})")
            historial += [Message(Role.USER, pregunta), Message(Role.ASSISTANT, final.text)]


if __name__ == "__main__":
    main()
