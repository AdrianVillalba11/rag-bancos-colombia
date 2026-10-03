"use strict";

/* Interfaz del asistente. Sin dependencias externas: todo el contenido dinámico se construye
   con la API del DOM (nunca con innerHTML), por lo que no hay inyección de HTML. */

const SESSION_KEY = "rag.sessionId";
const SESSIONS_KEY = "rag.sessions";
const MAX_SESSIONS = 50;
const SUGGESTIONS = [
  "¿Qué requisitos necesito para abrir una cuenta de ahorros?",
  "¿Qué tarjetas de crédito tienen cuota de manejo de $0?",
  "¿Cómo solicito un crédito de libre inversión?",
  "¿Qué es el 4x1000 y cómo se exonera?",
];

const els = {
  messages: document.getElementById("messages"),
  empty: document.getElementById("empty"),
  suggestions: document.getElementById("suggestions"),
  form: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  bank: document.getElementById("bank"),
  newChat: document.getElementById("new-chat"),
  notice: document.getElementById("notice"),
  counter: document.getElementById("counter"),
  sidebar: document.getElementById("sidebar"),
  backdrop: document.getElementById("backdrop"),
  historyList: document.getElementById("history-list"),
  historyEmpty: document.getElementById("history-empty"),
  toggleHistory: document.getElementById("toggle-history"),
  closeHistory: document.getElementById("close-history"),
};

const state = { sessionId: null, busy: false, maxLength: 1000 };

/* ---------- Utilidades ---------- */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function storage(action, value) {
  try {
    if (action === "get") return localStorage.getItem(SESSION_KEY);
    if (action === "set") localStorage.setItem(SESSION_KEY, value);
    if (action === "remove") localStorage.removeItem(SESSION_KEY);
  } catch (_) { /* almacenamiento no disponible: la sesión vive solo en memoria */ }
  return null;
}

/* Conversaciones de este navegador (el ID de sesión es la credencial de cada conversación). */
function loadSessionIds() {
  try {
    const ids = JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]");
    return Array.isArray(ids) ? ids.filter((x) => typeof x === "string") : [];
  } catch (_) { return []; }
}

function saveSessionIds(ids) {
  try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(ids.slice(0, MAX_SESSIONS))); }
  catch (_) { /* almacenamiento no disponible */ }
}

function rememberSession(id) {
  const ids = loadSessionIds().filter((x) => x !== id);
  ids.unshift(id);
  saveSessionIds(ids);
}

function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch (_) { return null; }
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function showNotice(text) {
  els.notice.textContent = text;
  els.notice.hidden = !text;
}

/* ---------- Render de texto (subconjunto seguro de Markdown) ---------- */
function appendInline(parent, text) {
  // **negrita** y marcas de cita [1]
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d{1,2}\])/g);
  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      parent.appendChild(el("strong", "", part.slice(2, -2)));
    } else if (/^\[\d{1,2}\]$/.test(part)) {
      parent.appendChild(el("span", "cite", part));
    } else {
      parent.appendChild(document.createTextNode(part));
    }
  }
}

function renderText(container, text) {
  container.replaceChildren();
  let list = null;
  let listType = null;
  let paragraph = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const p = el("p");
    appendInline(p, paragraph.join(" "));
    container.appendChild(p);
    paragraph = [];
  };

  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[*\-•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (!list || listType !== type) {
        list = el(type);
        listType = type;
        container.appendChild(list);
      }
      const li = el("li");
      appendInline(li, (bullet || numbered)[1]);
      list.appendChild(li);
    } else if (!line.trim()) {
      flushParagraph();
      list = null;
    } else {
      list = null;
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
}

/* ---------- Mensajes ---------- */
function hideEmpty() { els.empty.hidden = true; }

function addUserMessage(text) {
  hideEmpty();
  const wrap = el("div", "msg user");
  wrap.appendChild(el("div", "bubble", text));
  els.messages.appendChild(wrap);
  scrollToBottom();
}

function createAssistantMessage() {
  hideEmpty();
  const wrap = el("div", "msg assistant");
  const bubble = el("div", "bubble");
  const typing = el("div", "typing");
  typing.setAttribute("aria-label", "Escribiendo");
  typing.append(el("span"), el("span"), el("span"));
  bubble.appendChild(typing);
  wrap.appendChild(bubble);
  els.messages.appendChild(wrap);
  scrollToBottom();
  return { wrap, bubble, text: "" };
}

function renderSources(wrap, citations) {
  if (!citations || !citations.length) return;
  const box = el("div", "sources");
  box.appendChild(el("h3", "", "Fuentes"));
  const ol = el("ol");
  for (const c of citations) {
    const href = safeUrl(c.url);
    if (!href) continue;
    const li = el("li");
    li.appendChild(el("span", "tag", c.bank));
    const a = el("a", "", c.title || href);
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    li.appendChild(a);
    ol.appendChild(li);
  }
  if (ol.children.length) {
    box.appendChild(ol);
    wrap.appendChild(box);
  }
}

function renderFeedback(wrap, messageId, current, latencyMs) {
  const bar = el("div", "actions");
  const mk = (value, label, title) => {
    const b = el("button", "icon-btn", label);
    b.type = "button";
    b.title = title;
    b.setAttribute("aria-label", title);
    b.setAttribute("aria-pressed", String(current === value));
    b.addEventListener("click", async () => {
      try {
        const r = await fetch(`/api/messages/${messageId}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value }),
        });
        if (!r.ok) return;
        for (const other of bar.querySelectorAll("button")) {
          other.setAttribute("aria-pressed", String(other === b));
        }
      } catch (_) { /* sin conexión: no se marca */ }
    });
    return b;
  };
  bar.append(mk(1, "👍", "Respuesta útil"), mk(-1, "👎", "Respuesta no útil"));
  if (latencyMs) bar.appendChild(el("span", "meta", `${(latencyMs / 1000).toFixed(1)} s`));
  wrap.appendChild(bar);
}

function showError(msg, message) {
  msg.bubble.classList.add("error");
  msg.bubble.textContent = message || "Ocurrió un error. Inténtalo de nuevo.";
}

/* ---------- Streaming SSE ---------- */
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

async function ask(question) {
  if (state.busy) return;
  state.busy = true;
  setBusy(true);
  addUserMessage(question);
  const msg = createAssistantMessage();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: question,
        session_id: state.sessionId,
        bank: els.bank.value || null,
        stream: true,
      }),
    });

    if (!response.ok) {
      let message = null;
      try { message = (await response.json()).error.message; } catch (_) { /* sin cuerpo */ }
      showError(msg, message);
      return;
    }

    for await (const { event, data } of readEvents(response)) {
      if (event === "meta") {
        state.sessionId = data.session_id;
        storage("set", data.session_id);
        rememberSession(data.session_id);
      } else if (event === "token") {
        msg.text += data.text;
        renderText(msg.bubble, msg.text);
        scrollToBottom();
      } else if (event === "done") {
        renderSources(msg.wrap, data.citations);
        renderFeedback(msg.wrap, data.message_id, null, data.latency_ms);
      } else if (event === "error") {
        showError(msg, data.message);
      }
    }
  } catch (_) {
    showError(msg, "No se pudo conectar con el servicio. Revisa tu conexión e inténtalo de nuevo.");
  } finally {
    state.busy = false;
    setBusy(false);
    refreshHistory();
    scrollToBottom();
    els.input.focus();
  }
}

function setBusy(busy) {
  els.send.disabled = busy;
  els.input.disabled = busy;
  els.send.textContent = busy ? "Pensando…" : "Enviar";
}

/* ---------- Historial de conversaciones ---------- */
const dateFormat = new Intl.DateTimeFormat("es-CO", {
  day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
});

function setDrawer(open) {
  els.sidebar.classList.toggle("open", open);
  els.backdrop.hidden = !open;
}

async function refreshHistory() {
  const ids = loadSessionIds();
  els.historyList.replaceChildren();
  els.historyEmpty.hidden = ids.length > 0;
  if (!ids.length) return;
  try {
    const r = await fetch(`/api/sessions?ids=${encodeURIComponent(ids.join(","))}`);
    if (!r.ok) return;
    const sessions = await r.json();
    // Se descartan los ID que ya no existen en el servidor
    saveSessionIds(ids.filter((id) => sessions.some((s) => s.session_id === id)));
    els.historyEmpty.hidden = sessions.length > 0;
    for (const s of sessions) els.historyList.appendChild(historyItem(s));
  } catch (_) { /* sin conexión: se conserva lo mostrado */ }
}

function historyItem(s) {
  const li = el("li", "history-item");
  if (s.session_id === state.sessionId) li.classList.add("active");

  const open = el("button", "history-open");
  open.type = "button";
  open.title = s.title;
  open.appendChild(el("span", "history-title", s.title || "Conversación sin título"));
  const when = dateFormat.format(new Date(s.last_activity_at));
  open.appendChild(el("span", "history-meta", `${when} · ${s.messages} mensajes`));
  open.addEventListener("click", () => openSession(s.session_id));

  const remove = el("button", "history-remove", "✕");
  remove.type = "button";
  remove.title = "Quitar de este navegador";
  remove.setAttribute("aria-label", `Quitar "${s.title}" del historial`);
  remove.addEventListener("click", () => {
    saveSessionIds(loadSessionIds().filter((id) => id !== s.session_id));
    if (state.sessionId === s.session_id) newConversation();
    refreshHistory();
  });

  li.append(open, remove);
  return li;
}

function clearMessages() {
  for (const node of [...els.messages.children]) {
    if (node !== els.empty) node.remove();
  }
  els.empty.hidden = false;
}

async function openSession(id) {
  if (state.busy) return;
  state.sessionId = id;
  storage("set", id);
  clearMessages();
  await loadHistory();
  setDrawer(false);
  refreshHistory();
}

/* ---------- Sesión, configuración y arranque ---------- */
async function loadHistory() {
  if (!state.sessionId) return;
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/messages`);
    if (!r.ok) throw new Error("sin historial");
    const messages = await r.json();
    for (const m of messages) {
      if (m.role === "user") {
        addUserMessage(m.content);
      } else {
        const msg = createAssistantMessage();
        renderText(msg.bubble, m.content);
        renderSources(msg.wrap, m.citations);
        renderFeedback(msg.wrap, m.id, m.feedback, null);
      }
    }
  } catch (_) {
    state.sessionId = null;
    storage("remove");
  }
}

function newConversation() {
  if (state.busy) return;
  state.sessionId = null;
  storage("remove");
  clearMessages();
  setDrawer(false);
  refreshHistory();
  els.input.focus();
}

function autoResize() {
  if (!els.input.value) {
    els.input.style.height = "";  // vuelve a la altura de una línea definida por CSS
  } else {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
  }
  els.counter.textContent = els.input.value.length > state.maxLength * 0.8
    ? `${els.input.value.length}/${state.maxLength}` : "";
}

async function init() {
  for (const text of SUGGESTIONS) {
    const chip = el("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => ask(text));
    els.suggestions.appendChild(chip);
  }

  try {
    const cfg = await (await fetch("/api/config")).json();
    state.maxLength = cfg.max_question_length;
    els.input.maxLength = cfg.max_question_length;
    for (const b of cfg.banks) {
      const option = el("option", "", b.name);
      option.value = b.id;
      els.bank.appendChild(option);
    }
  } catch (_) {
    showNotice("No se pudo cargar la configuración del servicio.");
  }

  fetch("/api/health").then((r) => r.json()).then((h) => {
    if (h && h.warm === false) {
      showNotice("Los modelos se están cargando; la primera respuesta puede tardar un poco.");
      setTimeout(() => showNotice(""), 20000);
    } else if (h && h.status !== "ok") {
      showNotice("Algunos servicios no están disponibles; las respuestas pueden fallar.");
    }
  }).catch(() => {});

  state.sessionId = storage("get");
  if (state.sessionId && !loadSessionIds().includes(state.sessionId)) {
    rememberSession(state.sessionId);  // migra la sesión guardada por versiones anteriores
  }
  await loadHistory();
  refreshHistory();

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = els.input.value.trim();
    if (!question) return;
    els.input.value = "";
    autoResize();
    ask(question);
  });
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.input.addEventListener("input", autoResize);
  els.newChat.addEventListener("click", newConversation);
  els.toggleHistory.addEventListener("click", () => setDrawer(true));
  els.closeHistory.addEventListener("click", () => setDrawer(false));
  els.backdrop.addEventListener("click", () => setDrawer(false));
  els.input.focus();
}

init();
