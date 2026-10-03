"""Presentación del informe de analítica en texto (consola o Markdown)."""

from __future__ import annotations

from typing import Any


def _pct(valor: float | None) -> str:
    return "sin datos" if valor is None else f"{valor * 100:.1f} %"


def _seg(ms: float) -> str:
    return f"{ms / 1000:.1f} s"


def render_markdown(r: dict[str, Any]) -> str:
    o, v, t = r["overview"], r["volume"], r["topics"]
    lat, g, s = r["latency"], r["knowledge_gaps"], r["sources"]
    f, c = r["feedback"], r["conversations"]
    supuesto = r["assumptions"]["manual_search_minutes"]

    L: list[str] = ["# Informe de analítica del asistente", ""]
    periodo = r["period_start"] or "todo el histórico"
    L.append(f"Generado: {r['generated_at']} · Periodo desde: {periodo}")

    L += ["", "## Valores de impacto", ""]
    L += [
        f"- Conversaciones: **{o['conversations']}** · Preguntas: **{o['questions']}**",
        "- Tasa de resolución (respuestas con información relevante): "
        f"**{_pct(o['resolution_rate'])}**",
        f"- Satisfacción (👍 sobre valoradas): **{_pct(o['satisfaction_rate'])}**",
        f"- Latencia media: **{o['avg_latency_seconds']} s**",
        f"- Tiempo estimado ahorrado: **{o['estimated_minutes_saved']} min** "
        f"(supuesto: {supuesto} min de búsqueda manual por consulta resuelta)",
    ]

    L += ["", "## 1. Volumen de uso", ""]
    L += [
        f"- Preguntas por conversación: {v['questions_per_conversation']}",
        f"- Respuestas generadas: {v['answers']}",
    ]
    for d in v["per_day"]:
        L.append(f"  - {d['date']}: {d['questions']} preguntas")

    L += ["", "## 2. Temas y preguntas frecuentes", ""]
    for x in t["by_topic"]:
        L.append(f"- {x['topic']}: {x['questions']} ({_pct(x['share'])})")
    terminos = ", ".join(f"{x['term']} ({x['count']})" for x in t["top_terms"])
    L += ["", f"Términos más consultados: {terminos}"]
    L += ["", "Preguntas más repetidas:"]
    for x in t["frequent_questions"]:
        L.append(f"- ({x['count']}) {x['question']}")

    L += ["", "## 3. Latencia", ""]
    L += [
        f"- Media {_seg(lat['avg_ms'])} · p50 {_seg(lat['p50_ms'])} · p90 {_seg(lat['p90_ms'])} "
        f"· p95 {_seg(lat['p95_ms'])} · máx {_seg(lat['max_ms'])} ({lat['samples']} respuestas)",
        "- Por etapa (media): "
        + " · ".join(f"{k} {_seg(x)}" for k, x in lat["avg_by_stage_ms"].items()),
    ]

    L += ["", "## 4. Huecos de conocimiento", ""]
    L.append(f"Total: {g['total']} " + str({k: n for k, n in g["by_reason"].items()}))
    for x in g["recent"]:
        L.append(f"- [{x['reason_label']}] {x['question']} (relevancia {x['top_score']})")

    L += ["", "## 5. Bancos y páginas más citadas", ""]
    citas = ", ".join(f"{x['bank']} ({x['citations']})" for x in s["citations_by_bank"])
    filtros = ", ".join(f"{x['bank']} ({x['questions']})" for x in s["filter_usage"])
    L.append(f"Citas por banco: {citas}")
    L.append(f"Uso del filtro: {filtros}")
    for x in s["top_pages"]:
        L.append(f"- ({x['citations']}) {x['title']} — {x['url']}")

    L += ["", "## 6. Feedback de usuarios", ""]
    L += [
        f"- 👍 {f['positive']} · 👎 {f['negative']} · cobertura {_pct(f['coverage'])} "
        f"· satisfacción {_pct(f['satisfaction_rate'])}",
    ]
    for q in f["negative_questions"]:
        L.append(f"- Valorada negativamente: {q}")

    L += ["", "## 7. Perfil de las conversaciones", ""]
    L += [
        f"- Mensajes por conversación: media {c['avg_messages']}, mediana {c['median_messages']}",
        f"- Conversaciones con preguntas de seguimiento: {_pct(c['follow_up_rate'])}",
        "- Conversaciones de una sola pregunta (posible abandono): "
        f"{_pct(c['single_question_rate'])}",
        f"- Seguimientos reescritos con el historial: {c['rewritten_follow_ups']} "
        f"de {c['follow_up_questions']}",
    ]
    return "\n".join(L) + "\n"
