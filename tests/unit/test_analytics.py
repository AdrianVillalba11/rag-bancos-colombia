from datetime import UTC, datetime, timedelta

import pytest

from rag.analytics.metrics import (
    build_turns,
    classify_topics,
    compute_report,
    fingerprint,
    percentile,
)
from rag.analytics.report import render_markdown
from rag.domain.models import Citation, Role, StoredMessage

T0 = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
_id = iter(range(1, 10_000))


def pregunta(sesion, texto, banco=None, minutos=0, dia=0):
    return StoredMessage(
        id=next(_id), session_id=sesion, role=Role.USER, content=texto,
        created_at=T0 + timedelta(days=dia, minutes=minutos), bank_filter=banco,
    )  # fmt: skip


def respuesta(sesion, texto="ok", *, answered=True, score=0.9, ms=2000, citas=(), feedback=None,
              reescrita=None, dia=0, minutos=1):  # fmt: skip
    return StoredMessage(
        id=next(_id), session_id=sesion, role=Role.ASSISTANT, content=texto,
        created_at=T0 + timedelta(days=dia, minutes=minutos), latency_ms=ms, retrieval_ms=ms // 10,
        rerank_ms=ms // 5, generation_ms=ms // 2, top_score=score, answered=answered,
        rewritten_query=reescrita, citations=tuple(citas), feedback=feedback,
    )  # fmt: skip


CITA_A = Citation("Cuenta de Ahorros", "https://x.com/a", "davivienda")
CITA_B = Citation("Tarjeta Libre", "https://x.com/b", "bancolombia")


def historico():
    return [
        # s1: tres turnos, con seguimiento reescrito y feedback positivo
        pregunta("s1", "¿Qué requisitos necesito para abrir una cuenta de ahorros?", "davivienda"),
        respuesta("s1", ms=3000, citas=[CITA_A], feedback=1),
        pregunta("s1", "¿Y cuánto cuesta la cuota de manejo?", "davivienda", 2),
        respuesta("s1", ms=1000, citas=[CITA_A], reescrita="cuota de manejo cuenta", feedback=1),
        # s2: una sola pregunta, respondida, valorada negativamente
        pregunta("s2", "¿Qué tarjetas de crédito tienen cuota de manejo de $0?", None, 5, dia=1),
        respuesta("s2", ms=5000, citas=[CITA_B], feedback=-1, dia=1, minutos=6),
        # s3: sin respuesta (fuera de tema)
        pregunta("s3", "¿Quién ganó el mundial?", None, 10, dia=1),
        respuesta("s3", answered=False, score=0.02, ms=500, dia=1, minutos=11),
        # s4: respondida con baja relevancia, misma pregunta repetida de s1 (variando formato)
        pregunta("s4", "que requisitos necesito para abrir una CUENTA de ahorros", None, 20, dia=1),
        respuesta("s4", score=0.3, ms=4000, citas=[CITA_A], dia=1, minutos=21),
    ]


# --- Utilidades -------------------------------------------------------------------------------
def test_fingerprint_ignora_acentos_mayusculas_y_signos():
    assert fingerprint("¿Qué es el 4x1000?") == fingerprint("que es el 4X1000")


def test_percentiles_por_rango_mas_cercano():
    assert percentile([], 90) == 0.0
    assert percentile([10], 95) == 10
    datos = list(range(1, 101))
    assert (
        percentile(datos, 50) == 50 and percentile(datos, 90) == 90 and percentile(datos, 95) == 95
    )


def test_los_turnos_emparejan_pregunta_y_respuesta_por_sesion():
    msgs = [
        pregunta("a", "p1"), pregunta("b", "q1"), respuesta("b"), respuesta("a"),
        pregunta("a", "p2"),  # sin respuesta
    ]  # fmt: skip
    turnos, por_sesion = build_turns(msgs)
    assert [(t.session_id, t.question.content, t.answer is not None) for t in turnos] == [
        ("b", "q1", True), ("a", "p1", True), ("a", "p2", False)
    ]  # fmt: skip
    assert [t.position for t in turnos if t.session_id == "a"] == [0, 1]
    assert por_sesion == {"a": 3, "b": 2}


@pytest.mark.parametrize(
    ("texto", "tema"),
    [
        (
            "¿Qué requisitos necesito para abrir una cuenta de ahorros?",
            "Cuentas de ahorro y nómina",
        ),
        ("¿Qué tarjetas de crédito tienen cuota de manejo?", "Tarjetas"),
        ("¿Cómo solicito un crédito de libre inversión?", "Créditos y vivienda"),
        ("¿Qué es el 4x1000 y cómo se exonera?", "Impuestos (4x1000)"),
        ("¿Qué es un CDT?", "Inversión y ahorro programado"),
        ("¿Quién ganó el mundial?", "Otros"),
    ],
)
def test_clasificacion_de_temas(texto, tema):
    assert tema in classify_topics(texto)


def test_libre_inversion_es_un_credito_y_no_una_inversion():
    temas = classify_topics("crédito de libre inversión")
    assert "Créditos y vivienda" in temas and "Inversión y ahorro programado" not in temas


# --- Informe completo -------------------------------------------------------------------------
def test_valores_de_impacto():
    o = compute_report(historico(), manual_minutes=5)["overview"]
    assert o["conversations"] == 4 and o["questions"] == 5
    assert o["resolution_rate"] == pytest.approx(0.8, abs=1e-3)  # 4 de 5 respondidas
    assert o["satisfaction_rate"] == pytest.approx(2 / 3, abs=1e-3)  # 2 positivas de 3 valoradas
    assert o["avg_latency_seconds"] == pytest.approx(2.7)
    assert o["estimated_minutes_saved"] == 20  # 4 respondidas x 5 min


def test_volumen_y_serie_diaria():
    v = compute_report(historico())["volume"]
    assert v["questions"] == 5 and v["answers"] == 5
    assert v["questions_per_conversation"] == 1.25
    assert v["per_day"] == [
        {"date": "2026-10-01", "questions": 2}, {"date": "2026-10-02", "questions": 3}
    ]  # fmt: skip


def test_temas_terminos_y_preguntas_frecuentes():
    t = compute_report(historico())["topics"]
    assert any(
        x["topic"] == "Cuentas de ahorro y nómina" and x["questions"] >= 2 for x in t["by_topic"]
    )
    assert t["frequent_questions"][0]["count"] == 2  # la pregunta repetida, con otro formato
    assert "banco" not in {x["term"] for x in t["top_terms"]}


def test_latencia_con_percentiles_y_etapas():
    lat = compute_report(historico())["latency"]
    assert lat["samples"] == 5 and lat["max_ms"] == 5000
    assert lat["avg_ms"] == 2700 and lat["p50_ms"] == 3000
    assert lat["avg_by_stage_ms"]["generation"] == 1350  # mitad de la latencia media


def test_huecos_de_conocimiento_con_su_razon():
    g = compute_report(historico())["knowledge_gaps"]
    razones = {x["question"]: x["reason"] for x in g["recent"]}
    assert razones["¿Quién ganó el mundial?"] == "sin_respuesta"
    assert razones["¿Qué tarjetas de crédito tienen cuota de manejo de $0?"] == "feedback_negativo"
    assert razones["que requisitos necesito para abrir una CUENTA de ahorros"] == "baja_relevancia"
    assert g["total"] == 3 and g["by_reason"]["sin_respuesta"] == 1


def test_bancos_y_paginas_citadas():
    s = compute_report(historico())["sources"]
    assert s["citations_by_bank"][0] == {"bank": "davivienda", "citations": 3}
    assert s["top_pages"][0]["url"] == "https://x.com/a" and s["top_pages"][0]["citations"] == 3
    assert {x["bank"]: x["questions"] for x in s["filter_usage"]} == {"davivienda": 2, "todos": 3}


def test_feedback():
    f = compute_report(historico())["feedback"]
    assert (f["positive"], f["negative"], f["rated"]) == (2, 1, 3)
    assert f["coverage"] == pytest.approx(3 / 5, abs=1e-3) and f[
        "satisfaction_rate"
    ] == pytest.approx(2 / 3, abs=1e-3)
    assert f["negative_questions"] == ["¿Qué tarjetas de crédito tienen cuota de manejo de $0?"]


def test_perfil_de_conversaciones():
    c = compute_report(historico())["conversations"]
    assert c["total"] == 4
    assert c["follow_up_rate"] == 0.25 and c["single_question_rate"] == 0.75
    assert c["follow_up_questions"] == 1 and c["rewritten_follow_ups"] == 1
    assert c["avg_messages"] == 2.5


def test_el_filtro_de_periodo_excluye_lo_anterior():
    desde = T0 + timedelta(days=1)
    r = compute_report(historico(), since=desde)
    assert r["overview"]["conversations"] == 3 and r["period_start"] is not None


def test_un_historico_vacio_no_falla_y_no_inventa_valores():
    r = compute_report([])
    assert r["overview"]["questions"] == 0 and r["overview"]["resolution_rate"] is None
    assert r["latency"]["samples"] == 0 and r["knowledge_gaps"]["recent"] == []
    assert "sin datos" in render_markdown(r)


def test_el_informe_en_texto_incluye_las_siete_secciones():
    texto = render_markdown(compute_report(historico()))
    for titulo in ("Valores de impacto", "1. Volumen", "2. Temas", "3. Latencia",
                   "4. Huecos", "5. Bancos", "6. Feedback", "7. Perfil"):  # fmt: skip
        assert titulo in texto
