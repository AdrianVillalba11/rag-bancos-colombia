"""Análisis del histórico de conversaciones.

`compute_report` recorre los mensajes una sola vez, los empareja en turnos (pregunta -> respuesta)
y calcula métricas de uso, calidad e impacto. Es una función pura: no toca la base de datos, lo que
permite probarla con datos sintéticos y reutilizarla desde la API, el dashboard y el script.
"""

from __future__ import annotations

import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rag.domain.models import Role, StoredMessage
from rag.retrieval.lexical import tokenize

#: Temas del dominio y prefijos de términos que los identifican (sin acentos, en minúscula).
#: Es una clasificación por reglas: simple, explicable y sin costo de modelo.
TOPICS: dict[str, tuple[str, ...]] = {
    "Cuentas de ahorro y nómina": ("cuent", "ahorr", "nomin", "pension", "debito"),
    "Tarjetas": ("tarjet", "visa", "mastercard", "amex", "american"),
    "Créditos y vivienda": (
        "credit", "prest", "libranz", "hipotec", "vivienda", "vehicul", "libre", "leasing",
    ),
    "Tarifas y costos": (
        "cuota", "tarifa", "costo", "cobr", "comision", "interes", "tasa", "precio",
    ),
    "Requisitos y trámites": ("requisit", "document", "tramit", "abrir", "solicit", "pasos"),
    "Seguros": ("seguro", "poliza", "asistencia"),
    "Inversión y ahorro programado": ("cdt", "cdat", "fondo", "invert", "inversion", "dafuturo"),
    "Canales y atención": ("app", "oficin", "cajero", "sucursal", "linea", "telefon", "pse", "web"),
    "Impuestos (4x1000)": ("4x1000", "gmf", "impuest", "exoner"),
}
OTROS = "Otros"

#: Palabras demasiado genéricas para ser un término informativo de las preguntas
_TERMINOS_GENERICOS = frozenset(
    "banco bancos quiero puedo puede necesito saber informacion hacer tiene tienen ofrece "
    "ofrecen existe existen cuales cuanto cuantos como cual que quien donde".split()
)

_RAZONES = {
    "sin_respuesta": "No se encontró información relevante",
    "baja_relevancia": "Respondida con baja relevancia",
    "feedback_negativo": "Marcada como no útil por el usuario",
}


@dataclass(frozen=True)
class Turn:
    """Una pregunta del usuario y, si existe, su respuesta."""

    session_id: str
    position: int  # 0 = primera pregunta de la conversación
    question: StoredMessage
    answer: StoredMessage | None


def fingerprint(texto: str) -> str:
    """Clave para agrupar preguntas iguales ignorando mayúsculas, acentos y puntuación."""
    base = unicodedata.normalize("NFKD", texto.lower())
    base = "".join(c for c in base if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in base).split())


def percentile(valores: list[float], p: float) -> float:
    """Percentil por rango más cercano (suficiente para métricas operativas)."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    indice = max(0, min(len(ordenados) - 1, math.ceil(p / 100 * len(ordenados)) - 1))
    return float(ordenados[indice])


def _media(valores: list[float]) -> float:
    return round(statistics.fmean(valores), 1) if valores else 0.0


def _tasa(parte: int, total: int) -> float | None:
    return round(parte / total, 4) if total else None


def build_turns(messages: Iterable[StoredMessage]) -> tuple[list[Turn], dict[str, int]]:
    """Empareja cada pregunta con la respuesta que le sigue en su misma sesión.

    Devuelve los turnos y el número de mensajes por sesión.
    """
    pendiente: dict[str, StoredMessage] = {}
    contador: dict[str, int] = defaultdict(int)
    mensajes_por_sesion: dict[str, int] = defaultdict(int)
    turnos: list[Turn] = []

    def cerrar(sid: str, respuesta: StoredMessage | None) -> None:
        pregunta = pendiente.pop(sid)
        turnos.append(Turn(sid, contador[sid], pregunta, respuesta))
        contador[sid] += 1

    for m in messages:
        mensajes_por_sesion[m.session_id] += 1
        if m.role == Role.USER:
            if m.session_id in pendiente:  # pregunta anterior sin respuesta (fallo o abandono)
                cerrar(m.session_id, None)
            pendiente[m.session_id] = m
        elif m.session_id in pendiente:
            cerrar(m.session_id, m)
    for sid in list(pendiente):
        cerrar(sid, None)
    return turnos, dict(mensajes_por_sesion)


def classify_topics(pregunta: str) -> list[str]:
    tokens = tokenize(pregunta)
    if "libre" in tokens:  # "libre inversión" es un crédito, no una inversión
        tokens = [t for t in tokens if t != "inversion"]
    # Se incluye también el texto normalizado completo para cifras como "4x1000"
    texto = fingerprint(pregunta).replace(" ", "")
    encontrados = [
        tema
        for tema, prefijos in TOPICS.items()
        if any(t.startswith(p) for t in tokens for p in prefijos)
        or any(p in texto for p in prefijos if any(c.isdigit() for c in p))
    ]
    return encontrados or [OTROS]


def compute_report(
    messages: Iterable[StoredMessage],
    *,
    since: datetime | None = None,
    manual_minutes: float = 4.0,
    top_n: int = 10,
    low_relevance: float = 0.5,
) -> dict[str, Any]:
    """Calcula el informe completo a partir del histórico de mensajes."""
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    filtrados = [m for m in messages if since is None or m.created_at >= since]
    turnos, mensajes_por_sesion = build_turns(filtrados)

    respuestas = [t.answer for t in turnos if t.answer is not None]
    contestadas = [a for a in respuestas if a.answered]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "period_start": since.isoformat() if since else None,
        "assumptions": {"manual_search_minutes": manual_minutes, "low_relevance": low_relevance},
        "overview": _overview(turnos, respuestas, contestadas, mensajes_por_sesion, manual_minutes),
        "volume": _volume(turnos, mensajes_por_sesion),
        "topics": _topics(turnos, top_n),
        "latency": _latency(respuestas),
        "knowledge_gaps": _gaps(turnos, low_relevance, top_n * 2),
        "sources": _sources(turnos, respuestas, top_n),
        "feedback": _feedback(turnos, respuestas),
        "conversations": _conversations(turnos, mensajes_por_sesion),
    }


# --- Bloques ----------------------------------------------------------------------------------
def _overview(
    turnos: list[Turn],
    respuestas: list[StoredMessage],
    contestadas: list[StoredMessage],
    por_sesion: dict[str, int],
    manual_minutes: float,
) -> dict[str, Any]:
    valoradas = [a for a in respuestas if a.feedback is not None]
    positivas = [a for a in valoradas if a.feedback == 1]
    latencias = [a.latency_ms for a in respuestas if a.latency_ms is not None]
    return {
        "conversations": len(por_sesion),
        "questions": len(turnos),
        "resolution_rate": _tasa(len(contestadas), len(respuestas)),
        "satisfaction_rate": _tasa(len(positivas), len(valoradas)),
        "avg_latency_seconds": round(_media([float(x) for x in latencias]) / 1000, 2),
        "estimated_minutes_saved": round(len(contestadas) * manual_minutes, 1),
    }


def _volume(turnos: list[Turn], por_sesion: dict[str, int]) -> dict[str, Any]:
    por_dia: Counter[str] = Counter(t.question.created_at.date().isoformat() for t in turnos)
    return {
        "conversations": len(por_sesion),
        "questions": len(turnos),
        "answers": sum(1 for t in turnos if t.answer is not None),
        "questions_per_conversation": round(len(turnos) / len(por_sesion), 2) if por_sesion else 0,
        "per_day": [{"date": d, "questions": n} for d, n in sorted(por_dia.items())],
    }


def _topics(turnos: list[Turn], top_n: int) -> dict[str, Any]:
    temas: Counter[str] = Counter()
    terminos: Counter[str] = Counter()
    frecuentes: dict[str, dict[str, Any]] = {}
    for t in turnos:
        temas.update(classify_topics(t.question.content))
        terminos.update(x for x in tokenize(t.question.content) if x not in _TERMINOS_GENERICOS)
        clave = fingerprint(t.question.content)
        entrada = frecuentes.setdefault(clave, {"question": t.question.content, "count": 0})
        entrada["count"] += 1
    total = len(turnos) or 1
    return {
        "by_topic": [
            {"topic": tema, "questions": n, "share": round(n / total, 4)}
            for tema, n in temas.most_common()
        ],
        "top_terms": [{"term": x, "count": n} for x, n in terminos.most_common(top_n)],
        "frequent_questions": sorted(frecuentes.values(), key=lambda e: -e["count"])[:top_n],
    }


def _latency(respuestas: list[StoredMessage]) -> dict[str, Any]:
    def serie(attr: str) -> list[float]:
        return [float(getattr(a, attr)) for a in respuestas if getattr(a, attr) is not None]

    totales = serie("latency_ms")
    return {
        "samples": len(totales),
        "avg_ms": _media(totales),
        "p50_ms": percentile(totales, 50),
        "p90_ms": percentile(totales, 90),
        "p95_ms": percentile(totales, 95),
        "max_ms": max(totales) if totales else 0.0,
        "avg_by_stage_ms": {
            "retrieval": _media(serie("retrieval_ms")),
            "rerank": _media(serie("rerank_ms")),
            "generation": _media(serie("generation_ms")),
        },
    }


def _gaps(turnos: list[Turn], low_relevance: float, limit: int) -> dict[str, Any]:
    """Preguntas que el sistema no supo resolver bien: son el insumo para ampliar el contenido."""
    huecos: list[dict[str, Any]] = []
    for t in turnos:
        a = t.answer
        if a is None:
            continue
        if a.answered is False:
            razon = "sin_respuesta"
        elif a.feedback == -1:
            razon = "feedback_negativo"
        elif a.top_score is not None and a.top_score < low_relevance:
            razon = "baja_relevancia"
        else:
            continue
        huecos.append(
            {
                "question": t.question.content,
                "bank": t.question.bank_filter,
                "top_score": round(a.top_score, 3) if a.top_score is not None else None,
                "reason": razon,
                "reason_label": _RAZONES[razon],
                "date": t.question.created_at.date().isoformat(),
            }
        )
    por_razon = Counter(h["reason"] for h in huecos)
    return {
        "total": len(huecos),
        "by_reason": dict(por_razon),
        "recent": list(reversed(huecos))[:limit],
    }


def _sources(turnos: list[Turn], respuestas: list[StoredMessage], top_n: int) -> dict[str, Any]:
    por_banco: Counter[str] = Counter()
    por_url: Counter[str] = Counter()
    titulos: dict[str, str] = {}
    for a in respuestas:
        for c in a.citations:
            por_banco[c.bank] += 1
            por_url[c.url] += 1
            titulos.setdefault(c.url, c.title)
    filtro = Counter((t.question.bank_filter or "todos") for t in turnos)
    return {
        "citations_by_bank": [{"bank": b, "citations": n} for b, n in por_banco.most_common()],
        "top_pages": [
            {"url": u, "title": titulos[u], "citations": n} for u, n in por_url.most_common(top_n)
        ],
        "filter_usage": [{"bank": b, "questions": n} for b, n in filtro.most_common()],
    }


def _feedback(turnos: list[Turn], respuestas: list[StoredMessage]) -> dict[str, Any]:
    positivas = sum(1 for a in respuestas if a.feedback == 1)
    negativas = sum(1 for a in respuestas if a.feedback == -1)
    valoradas = positivas + negativas
    return {
        "positive": positivas,
        "negative": negativas,
        "rated": valoradas,
        "coverage": _tasa(valoradas, len(respuestas)),
        "satisfaction_rate": _tasa(positivas, valoradas),
        "negative_questions": [
            t.question.content for t in turnos if t.answer is not None and t.answer.feedback == -1
        ][:10],
    }


def _conversations(turnos: list[Turn], por_sesion: dict[str, int]) -> dict[str, Any]:
    preguntas: Counter[str] = Counter(t.session_id for t in turnos)
    n = len(por_sesion)
    con_seguimiento = sum(1 for c in preguntas.values() if c >= 2)
    una_sola = sum(1 for c in preguntas.values() if c == 1)
    seguimientos = [t for t in turnos if t.position > 0]
    reescritas = sum(1 for t in seguimientos if t.answer and t.answer.rewritten_query)
    longitudes = [float(x) for x in por_sesion.values()]
    return {
        "total": n,
        "avg_messages": _media(longitudes),
        "median_messages": statistics.median(longitudes) if longitudes else 0,
        "follow_up_rate": _tasa(con_seguimiento, n),
        "single_question_rate": _tasa(una_sola, n),
        "follow_up_questions": len(seguimientos),
        "rewritten_follow_ups": reescritas,
    }
