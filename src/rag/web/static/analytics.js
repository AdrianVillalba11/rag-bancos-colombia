"use strict";

/* Dashboard de analítica. Todo el contenido se construye con la API del DOM. */

const TOKEN_KEY = "rag.analyticsToken";
const els = {
  period: document.getElementById("period"),
  refresh: document.getElementById("refresh"),
  status: document.getElementById("status"),
  content: document.getElementById("content"),
  kpis: document.getElementById("kpis"),
  tokenForm: document.getElementById("token-form"),
  token: document.getElementById("token"),
  assumptions: document.getElementById("assumptions"),
};

function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

const pct = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)} %`);
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
const body = (id) => document.querySelector(`#${id} .body`);
const empty = (text) => el("p", "empty-note", text);

function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch (_) { return null; }
}

function getToken() { try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch (_) { return ""; } }
function setToken(v) { try { sessionStorage.setItem(TOKEN_KEY, v); } catch (_) { /* sin almacenamiento */ } }

/* ---------- Componentes ---------- */
function bars(rows, { warn = false } = {}) {
  const max = Math.max(1, ...rows.map((r) => r.value));
  const box = el("div", "bars");
  for (const r of rows) {
    const row = el("div", "bar-row");
    const label = el("span", "bar-label", r.label);
    label.title = r.label;
    const track = el("div", "bar-track");
    const fill = el("div", warn ? "bar-fill warn" : "bar-fill");
    fill.style.width = `${Math.max(2, (r.value / max) * 100)}%`;
    track.appendChild(fill);
    row.append(label, track, el("span", "bar-num", r.text ?? String(r.value)));
    box.appendChild(row);
  }
  return box;
}

function stat(value, label) {
  const s = el("div", "stat");
  s.append(el("div", "v", String(value)), el("div", "l", label));
  return s;
}

function kpi(value, label, hint) {
  const k = el("div", "kpi");
  k.append(el("div", "value", value), el("div", "label", label));
  if (hint) k.appendChild(el("div", "hint", hint));
  return k;
}

function fill(id, ...nodes) {
  const target = body(id);
  target.replaceChildren(...nodes);
}

/* ---------- Secciones ---------- */
function render(r) {
  const o = r.overview;
  els.kpis.replaceChildren(
    kpi(String(o.conversations), "Conversaciones"),
    kpi(String(o.questions), "Preguntas"),
    kpi(pct(o.resolution_rate), "Tasa de resolución", "Respuestas con información relevante"),
    kpi(pct(o.satisfaction_rate), "Satisfacción", "👍 sobre respuestas valoradas"),
    kpi(`${o.avg_latency_seconds} s`, "Latencia media"),
    kpi(`${o.estimated_minutes_saved} min`, "Tiempo ahorrado (estimado)",
      `${r.assumptions.manual_search_minutes} min de búsqueda manual por consulta resuelta`),
  );

  // 1. Volumen
  const v = r.volume;
  const maxDay = Math.max(1, ...v.per_day.map((d) => d.questions));
  const cols = el("div", "cols");
  for (const d of v.per_day) {
    const col = el("div", "col");
    col.appendChild(el("span", "col-n", String(d.questions)));
    const f = el("div", "col-fill");
    f.style.height = `${Math.max(3, (d.questions / maxDay) * 100)}%`;
    col.append(f, el("span", "col-d", d.date.slice(5)));
    cols.appendChild(col);
  }
  fill("volume",
    (() => { const s = el("div", "stats"); s.append(
      stat(v.conversations, "conversaciones"), stat(v.questions, "preguntas"),
      stat(v.answers, "respuestas"), stat(v.questions_per_conversation, "preguntas por conversación")); return s; })(),
    el("h3", "", "Preguntas por día"),
    v.per_day.length ? cols : empty("Sin datos en el periodo."));

  // 2. Temas, frecuentes y términos
  const t = r.topics;
  fill("topics", t.by_topic.length
    ? bars(t.by_topic.map((x) => ({ label: x.topic, value: x.questions, text: `${x.questions} · ${pct(x.share)}` })))
    : empty("Sin datos."));

  const freq = el("ul", "list");
  for (const q of t.frequent_questions) {
    const li = el("li");
    li.append(el("span", "pill", `${q.count}×`), el("span", "", q.question));
    freq.appendChild(li);
  }
  const terms = t.top_terms.length
    ? bars(t.top_terms.map((x) => ({ label: x.term, value: x.count })))
    : empty("Sin datos.");
  fill("frequent", el("h3", "", "Preguntas más repetidas"),
    t.frequent_questions.length ? freq : empty("Sin datos."), el("h3", "", "Términos más consultados"), terms);

  // 3. Latencia
  const l = r.latency;
  const st = el("div", "stats");
  st.append(stat(secs(l.avg_ms), "media"), stat(secs(l.p50_ms), "p50"), stat(secs(l.p90_ms), "p90"),
    stat(secs(l.p95_ms), "p95"), stat(secs(l.max_ms), "máxima"));
  const stages = l.avg_by_stage_ms;
  fill("latency", st, el("h3", "", "Tiempo medio por etapa"), bars([
    { label: "Recuperación", value: stages.retrieval, text: secs(stages.retrieval) },
    { label: "Reranker", value: stages.rerank, text: secs(stages.rerank) },
    { label: "Generación", value: stages.generation, text: secs(stages.generation) },
  ]));

  // 4. Huecos de conocimiento
  const g = r.knowledge_gaps;
  const gl = el("ul", "list");
  for (const x of g.recent) {
    const li = el("li");
    const cls = x.reason === "sin_respuesta" ? "pill bad" : "pill warn";
    li.append(el("span", cls, x.reason_label), el("span", "", x.question));
    gl.appendChild(li);
  }
  fill("gaps",
    el("p", "", `${g.total} preguntas con problemas de cobertura o calidad. Son la guía para ampliar el contenido indexado.`),
    g.recent.length ? gl : empty("No hay huecos detectados."));

  // 5. Fuentes
  const s = r.sources;
  const top = el("ul", "list");
  for (const p of s.top_pages) {
    const li = el("li");
    li.appendChild(el("span", "pill", `${p.citations}×`));
    const href = safeUrl(p.url);
    if (href) {
      const a = el("a", "", p.title || p.url);
      a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
      li.appendChild(a);
    } else { li.appendChild(el("span", "", p.title)); }
    top.appendChild(li);
  }
  fill("sources",
    el("h3", "", "Citas por banco"),
    s.citations_by_bank.length ? bars(s.citations_by_bank.map((x) => ({ label: x.bank, value: x.citations }))) : empty("Sin citas."),
    el("h3", "", "Uso del filtro de banco"),
    bars(s.filter_usage.map((x) => ({ label: x.bank, value: x.questions }))),
    el("h3", "", "Páginas más citadas"), s.top_pages.length ? top : empty("Sin citas."));

  // 6. Feedback
  const f = r.feedback;
  const fs = el("div", "stats");
  fs.append(stat(f.positive, "👍 útiles"), stat(f.negative, "👎 no útiles"),
    stat(pct(f.coverage), "respuestas valoradas"), stat(pct(f.satisfaction_rate), "satisfacción"));
  const neg = el("ul", "list");
  for (const q of f.negative_questions) {
    const li = el("li"); li.append(el("span", "pill bad", "👎"), el("span", "", q)); neg.appendChild(li);
  }
  fill("feedback", fs, ...(f.negative_questions.length ? [el("h3", "", "Valoradas negativamente"), neg] : []));

  // 7. Conversaciones
  const c = r.conversations;
  const cs = el("div", "stats");
  cs.append(stat(c.avg_messages, "mensajes por conversación (media)"), stat(c.median_messages, "mediana"),
    stat(pct(c.follow_up_rate), "con preguntas de seguimiento"), stat(pct(c.single_question_rate), "de una sola pregunta"));
  fill("conversations", cs,
    el("p", "", `Seguimientos reescritos con el historial: ${c.rewritten_follow_ups} de ${c.follow_up_questions}.`));

  els.assumptions.textContent =
    `Generado: ${new Date(r.generated_at).toLocaleString("es-CO")}. ` +
    "Los temas se asignan por reglas de palabras clave; el tiempo ahorrado es una estimación basada en el supuesto indicado.";
}

/* ---------- Carga ---------- */
function setStatus(text) { els.status.textContent = text; els.status.hidden = !text; }

async function load() {
  setStatus("Cargando…");
  const days = els.period.value;
  const headers = {};
  if (getToken()) headers["X-Analytics-Token"] = getToken();
  try {
    const r = await fetch(`/api/analytics${days ? `?days=${days}` : ""}`, { headers });
    if (r.status === 401) {
      els.content.hidden = true;
      els.tokenForm.hidden = false;
      setStatus(getToken() ? "Token incorrecto." : "");
      return;
    }
    if (!r.ok) throw new Error(String(r.status));
    els.tokenForm.hidden = true;
    render(await r.json());
    els.content.hidden = false;
    setStatus("");
  } catch (_) {
    setStatus("No se pudo cargar la analítica. Inténtalo de nuevo.");
  }
}

els.refresh.addEventListener("click", load);
els.period.addEventListener("change", load);
els.tokenForm.addEventListener("submit", (e) => {
  e.preventDefault();
  setToken(els.token.value.trim());
  load();
});
load();
