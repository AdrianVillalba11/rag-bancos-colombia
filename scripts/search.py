"""Consulta de recuperación por línea de comandos (sin LLM), útil para inspeccionar resultados.

Uso (dentro de Docker):
    docker compose run --rm ingest python scripts/search.py "¿cuánto cuesta la cuota de manejo?"
    docker compose run --rm ingest python scripts/search.py "CDT" --bank davivienda --mode vector
"""

import argparse
import time

from rag.config import get_settings
from rag.infra.logging import configure_logging
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.indexer import build_embedder, build_vector_store
from rag.retrieval.lexical import Bm25Index
from rag.retrieval.rerankers import RerankerFactory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--bank", help="Filtra por banco")
    parser.add_argument("--mode", choices=["vector", "hybrid", "rerank"], default="rerank")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    store = build_vector_store(settings)
    index = Bm25Index(store.list_chunks()) if args.mode != "vector" else None
    retriever = HybridRetriever(
        build_embedder(settings),
        store,
        index,
        top_k=settings.retrieval_top_k,
        rrf_k=settings.rrf_k,
        hybrid=args.mode != "vector",
    )

    t0 = time.perf_counter()
    candidates = retriever.retrieve(args.question, args.bank)
    t1 = time.perf_counter()
    results = candidates[: args.top]
    if args.mode == "rerank":
        results = RerankerFactory.create(settings).rerank(
            args.question, candidates[: settings.rerank_candidates], args.top
        )
    t2 = time.perf_counter()

    print(f"\n[{args.mode}] recuperación {t1 - t0:.2f}s | rerank {t2 - t1:.2f}s\n")
    for i, r in enumerate(results, 1):
        c = r.chunk
        sim = f"sim={r.similarity:.2f} " if r.similarity is not None else ""
        print(f"{i}. [{c.bank}] {sim}score={r.score:.3f} {c.title} — {c.heading}")
        print(f"   {c.url}")
        print(f"   {c.text[:160].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
