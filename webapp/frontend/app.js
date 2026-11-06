// ============================================================
// DOM references
// ============================================================
const $ = (id) => document.getElementById(id);
const recentsList = $("recentsList");
const toast = $("toast");
const toastText = $("toastText");
const toastUndo = $("toastUndo");
const newChatBtn = $("newChatBtn");

const chatScroll = $("chatScroll");
const chatThread = $("chatThread");
const chatScrollDown = $("chatScrollDown");
const chatTitle = $("chatTitle");
const clearChatBtn = $("clearChatBtn");
const queryInput = $("queryInput");
const sendBtn = $("sendBtn");
const modeSelect = $("modeSelect");
const modelSelect = $("modelSelect");
const chatNote = $("chatNote");

const imageScroll = $("imageScroll");
const imageThread = $("imageThread");
const imageScrollDown = $("imageScrollDown");
const imageTitle = $("imageTitle");
const imageClearAllBtn = $("imageClearAllBtn");
const imageFileInput = $("imageFileInput");
const attachBtn = $("attachBtn");
const imageChip = $("imageChip");
const imageChipThumb = $("imageChipThumb");
const imageChipName = $("imageChipName");
const clearImageBtn = $("clearImageBtn");
const imageQuestionInput = $("imageQuestionInput");
const imageSendBtn = $("imageSendBtn");
const imageCompareBtn = $("imageCompareBtn");

// ============================================================
// Small helpers
// ============================================================
function isNarrow() { return window.innerWidth <= 820; }
function truncate(s, n) { s = String(s || "").replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n - 1) + "\u2026" : s; }
function newId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

function scrollToBottom(scroller) { scroller.scrollTop = scroller.scrollHeight; }
function isNearBottom(scroller) { return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 160; }

/** Shows a "jump to latest" arrow whenever the user has scrolled away from the bottom. */
function wireScrollDown(scroller, button) {
  scroller.addEventListener("scroll", () => {
    button.hidden = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 240;
  });
  button.addEventListener("click", () => scrollToBottom(scroller));
}

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
}

/** Appends a message bubble to a thread and scrolls its scroll area to the bottom. */
function appendMessage(container, classes, text) {
  const div = document.createElement("div");
  div.className = `message ${classes}`;
  div.textContent = text;
  container.appendChild(div);
  const scroller = container.closest(".chat-scroll");
  if (scroller) scrollToBottom(scroller);
  return div;
}

function copyText(text, button) {
  const flash = () => {
    const old = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = old; }, 1200);
  };
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(flash, () => {});
    }
  } catch (e) { /* clipboard unavailable - ignore */ }
}

/** Row under an answer: Copy, and (image tab) Re-run. */
function makeActions(text, onRerun) {
  const row = document.createElement("div");
  row.className = "msg-actions";
  const copy = document.createElement("button");
  copy.className = "link-btn";
  copy.textContent = "Copy";
  copy.addEventListener("click", () => copyText(text, copy));
  row.appendChild(copy);
  if (onRerun) {
    const rerun = document.createElement("button");
    rerun.className = "link-btn rerun-btn";
    rerun.textContent = "\u21BB Re-run";
    rerun.title = "Ask the model again (use this only if you want a fresh answer)";
    rerun.addEventListener("click", onRerun);
    row.appendChild(rerun);
  }
  return row;
}

/**
 * Turns a caught fetch() exception into a readable message. A browser-level network
 * failure ("Failed to fetch" / "NetworkError..." / "Load failed") means the BACKEND
 * itself could not be reached (uvicorn stopped or crashed). Ollama being down is a
 * different case: the backend is up and reports it as an "Error: ..." message.
 */
function describeRequestError(e) {
  const msg = String((e && e.message) || e);
  if (e && e.name === "TypeError" && /fetch|network|load failed/i.test(msg)) {
    return "Cannot reach the backend server - is uvicorn still running? " +
           "(If only Ollama is down, you would see an 'Error: ...' message instead.)";
  }
  return `Request failed: ${msg}`;
}

function renderMarkdown(text) {
  // Protect LaTeX delimiters from markdown's own backslash-escape handling.
  // Markdown treats "\[" as an escape sequence and strips the backslash,
  // leaving plain "[" - which destroys the \[ ... \] delimiter KaTeX needs to
  // find afterward. Doubling the backslash here means markdown's unescaping
  // consumes one backslash and leaves exactly one behind, intact for KaTeX.
  // Code fences and inline code are split out first and left untouched: doubling
  // backslashes there would show up literally (e.g. a regex "\\[" would display as "\\\\[")
  // because markdown does not unescape inside code.
  const protectedText = (text || "")
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, i) => i % 2 === 1 ? part : part
      .replace(/\\\[/g, "\\\\[")
      .replace(/\\\]/g, "\\\\]")
      .replace(/\\\(/g, "\\\\(")
      .replace(/\\\)/g, "\\\\)"))
    .join("");
  try {
    return marked.parse(protectedText);
  } catch (e) {
    return text; // fall back to plain text if marked.js failed to load
  }
}

/**
 * Runs KaTeX's auto-render extension on a DOM element - must be called AFTER
 * the element's innerHTML has been set (KaTeX needs real DOM nodes to scan).
 * Recognizes \[ ... \] and \( ... \) (common in model output) plus $$ ... $$
 * and $ ... $ as a fallback. Fails silently if KaTeX didn't load - the plain
 * markdown text still displays either way, just unrendered as math.
 */
function renderMathIn(element) {
  if (typeof renderMathInElement !== "function") return;
  try {
    renderMathInElement(element, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  } catch (e) {
    console.warn("KaTeX rendering failed:", e);
  }
}

/**
 * Final render of a model response into a message div: markdown -> HTML, then KaTeX.
 * Adds the "rendered" class (switches off pre-wrap, which is only wanted while raw
 * text streams in). An empty response is shown explicitly rather than as a blank box -
 * Qwen-family models are known to sometimes return nothing.
 */
function setRenderedHtml(div, text) {
  if (!text || !text.trim()) {
    div.textContent = "(The model returned an empty response.)";
    div.classList.add("empty-response");
    return;
  }
  div.innerHTML = renderMarkdown(text);
  div.classList.add("rendered");
  renderMathIn(div);
}

/** Short label shown after the model name in a routing summary. */
function routingTag(r) {
  let tag = "";
  if (r.manual_override) tag += " [MANUAL]";
  else if (r.exploration_occurred) tag += " [EXPLORATION]";
  if (r.fallback_triggered) tag += " [LOW RAM]"; // no model fit the RAM budget - smallest one used
  return tag;
}

/**
 * Adds what the panel needs to explain models that were NOT eligible: which ones, how much RAM each
 * would need, and the budget. The router only scores the models that fit, so the rest are derived from
 * `excluded_by_hard_constraint` + the per-model RAM needs the backend reports in /api/status.
 * Works on the router's raw result AND on an already-saved (slim) record.
 */
function withExclusions(r) {
  if (Array.isArray(r.excluded)) return r; // saved record: already derived when it was stored
  const needs = (typeof lastStatus !== "undefined" && lastStatus && lastStatus.model_requirements_gb) || {};
  const budget = r.budget_gb !== undefined ? r.budget_gb : (r.resource_state ? r.resource_state.stress_adjusted_ram_gb : undefined);
  return Object.assign({}, r, {
    budget_gb: budget,
    excluded: (r.excluded_by_hard_constraint || []).map((m) => ({ model: m, need_gb: needs[m] !== undefined ? needs[m] : null })),
  });
}

/** The router's sentence "Among 1 models that fit the RAM budget" is wrong when it fell back - fix it for display. */
function displayReason(r) {
  const text = r.selection_reason || "";
  if (!r.fallback_triggered) return text;
  return text.replace(/Among \d+ models that fit the ([a-z-]+) RAM budget \(([^)]*)\),/,
                      "No model fit the $1 RAM budget ($2) - fell back to the smallest model.");
}

function excludedText(e, budget) {
  const hasBudget = budget !== undefined && budget !== null;
  if (e.need_gb !== null && e.need_gb !== undefined && hasBudget) return `not eligible - needs ~${e.need_gb} GB, only ${budget} GB available`;
  return hasBudget ? `not eligible - not enough RAM (budget ${budget} GB)` : "not eligible - not enough RAM";
}

/** Builds the collapsible routing-decision panel as real DOM, not a text blob. */
function buildRoutingPanel(raw) {
  const result = withExclusions(raw);
  const details = document.createElement("details");
  details.className = "routing-panel";

  const summary = document.createElement("summary");
  summary.textContent =
    `${result.category} -> ${result.selected_model}${routingTag(result)} ` +
    (result.actual_response_time_sec ? `(${result.actual_response_time_sec}s)` : "");
  details.appendChild(summary);

  if (result.fallback_triggered) {
    const warn = document.createElement("div");
    warn.className = "routing-warning";
    const b = result.budget_gb !== undefined && result.budget_gb !== null ? ` (${result.budget_gb} GB)` : "";
    warn.textContent = `\u26A0 No model fits the RAM budget${b}. The router fell back to the smallest model ` +
      `(${result.selected_model}) and it may be slow. Closing apps or browser tabs frees memory and brings the other models back.`;
    details.appendChild(warn);
  }

  if (result.scores_breakdown) {
    const bd = result.scores_breakdown;
    const excluded = new Map((result.excluded || []).map((e) => [e.model, e]));
    const order = (result.candidates_considered && result.candidates_considered.length)
      ? result.candidates_considered : Object.keys(bd);
    const models = order.concat(Object.keys(bd).filter((m) => !order.includes(m)));

    const table = document.createElement("table");
    table.className = "routing-table";
    for (const model of models) {
      const row = document.createElement("tr");
      const b = bd[model];
      if (b) {
        if (model === result.selected_model) {
          row.className = result.manual_override ? "manual" : (result.exploration_occurred ? "exploration" : "selected");
        }
        const warm = b.warm ? "WARM" : "cold";
        const latency = b.cold_start_overhead_applied > 0
          ? `${b.latency_raw_sec}s -> ${b.latency_adjusted_sec}s (+${b.cold_start_overhead_applied}s)`
          : `${b.latency_raw_sec}s`;
        const pick = (result.manual_override && model === result.router_choice) ? " (router's pick)" : "";
        const over = excluded.has(model) ? " - over the RAM budget" : ""; // only possible when you forced it
        row.innerHTML = `<td>${model}${pick} [${warm}]</td>` +
          `<td>score=${b.final_score.toFixed(3)} | quality=${b.quality_raw} | latency=${latency}${over}</td>`;
      } else if (excluded.has(model)) {
        row.className = "excluded";
        row.innerHTML = `<td>${model} [excluded]</td><td>${excludedText(excluded.get(model), result.budget_gb)}</td>`;
      } else {
        continue;
      }
      table.appendChild(row);
    }
    details.appendChild(table);
  }

  const reason = document.createElement("div");
  reason.className = "routing-reason";
  reason.textContent = displayReason(result);
  details.appendChild(reason);

  return details;
}

// ============================================================
// Saved conversations (localStorage) - survive a page refresh until deleted/cleared.
// Every storage call is wrapped: storage can be full, disabled, or blocked, and the app
// must keep working without it.
// ============================================================
const Store = {
  load(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) { return fallback; }
  },
  save(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; }
  },
};

const STORE_KEY = "adaptiveAI.v2";
const MAX_CONVERSATIONS = 40;
const MAX_CHAT_ENTRIES = 120;
const MAX_IMAGE_ENTRIES = 80;
const MAX_HISTORY_MESSAGES = 6; // 3 exchanges sent as context, same as main.py

// conversation = {id, type:"chat"|"image", title, created, updated, entries:[], history:[], currentImage:null|{...}}
let state = { conversations: [], activeChatId: null, activeImageId: null, mode: "balanced", model: "auto",
              sidebarCollapsed: null, lastTab: "chatTab" };
let currentTab = "chatTab";

function loadState() {
  const saved = Store.load(STORE_KEY, null);
  if (saved && Array.isArray(saved.conversations)) state = Object.assign(state, saved);
}

function saveState() {
  if (state.conversations.length > MAX_CONVERSATIONS) state.conversations.length = MAX_CONVERSATIONS;
  // storage full: drop the oldest conversations (never the newest) until it fits
  while (!Store.save(STORE_KEY, state) && state.conversations.length > 1) state.conversations.pop();
}

function getConvo(id) { return id ? state.conversations.find((c) => c.id === id) || null : null; }
function chatConvo() { return getConvo(state.activeChatId); }
function imgConvo() { return getConvo(state.activeImageId); }

function createConvo(type, title) {
  const now = Date.now();
  const convo = { id: newId(), type, title: truncate(title, 42) || "New chat", created: now, updated: now,
                  entries: [], history: [], currentImage: null };
  state.conversations.unshift(convo);
  return convo;
}

/** Marks a conversation as just used and moves it to the top of Recents. */
function touchConvo(convo) {
  convo.updated = Date.now();
  const i = state.conversations.indexOf(convo);
  if (i > 0) { state.conversations.splice(i, 1); state.conversations.unshift(convo); }
}

/** Keeps the newest `max` entries and never starts on an answer whose question was cut off. */
function trimEntries(entries, max) {
  const out = entries.length > max ? entries.slice(-max) : entries.slice();
  while (out.length && (out[0].type === "assistant" || out[0].type === "compare")) out.shift();
  return out;
}

/** Only what the routing panel needs - keeps stored entries small. */
function slimRouting(r) {
  const d = withExclusions(r);
  return {
    category: r.category,
    candidates_considered: r.candidates_considered || null,
    excluded: d.excluded,
    budget_gb: d.budget_gb === undefined ? null : d.budget_gb,
    fallback_triggered: !!r.fallback_triggered,
    selected_model: r.selected_model,
    exploration_occurred: !!r.exploration_occurred,
    actual_response_time_sec: r.actual_response_time_sec,
    scores_breakdown: r.scores_breakdown || null,
    selection_reason: r.selection_reason || "",
    manual_override: !!r.manual_override,
    router_choice: r.router_choice || null,
    forced_model_fits_ram: r.forced_model_fits_ram === undefined ? null : r.forced_model_fits_ram,
  };
}

// ============================================================
// Sidebar: recents, navigation, status
// ============================================================
function setSidebarCollapsed(collapsed) {
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  state.sidebarCollapsed = collapsed;
  saveState();
}
document.querySelectorAll(".sidebar-hide").forEach((b) => b.addEventListener("click", () => setSidebarCollapsed(true)));
document.querySelectorAll(".sidebar-show").forEach((b) => b.addEventListener("click", () => setSidebarCollapsed(false)));

function renderRecents() {
  recentsList.innerHTML = "";
  if (!state.conversations.length) {
    const empty = document.createElement("div");
    empty.className = "recents-empty";
    empty.textContent = "No saved conversations yet";
    recentsList.appendChild(empty);
    return;
  }
  state.conversations.forEach((c) => {
    const item = document.createElement("div");
    item.className = "recent-item";
    item.dataset.id = c.id;
    const isActive = (c.type === "chat" && c.id === state.activeChatId && currentTab === "chatTab") ||
                     (c.type === "image" && c.id === state.activeImageId && currentTab === "imageTab");
    if (isActive) item.classList.add("active");

    const icon = document.createElement("span");
    icon.className = "recent-icon";
    icon.textContent = c.type === "image" ? "\uD83D\uDDBC" : "\uD83D\uDCAC";
    const title = document.createElement("span");
    title.className = "recent-title";
    title.textContent = c.title;
    title.title = c.title;
    const del = document.createElement("button");
    del.className = "recent-del";
    del.title = "Delete this conversation";
    del.textContent = "\u2715";
    del.addEventListener("click", (ev) => { ev.stopPropagation(); deleteConvo(c.id, true); });

    item.append(icon, title, del);
    item.addEventListener("click", () => openConvo(c.id));
    recentsList.appendChild(item);
  });
}

function showTab(name) {
  currentTab = name;
  state.lastTab = name;
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === name));
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  if (name === "historyTab") loadHistory();
  if (name === "chatTab") scrollToBottom(chatScroll);   // hidden panels lose their scroll position
  if (name === "imageTab") scrollToBottom(imageScroll);
  if (isNarrow()) document.body.classList.add("sidebar-collapsed");
  renderRecents();
  saveState();
}
document.querySelectorAll(".nav-item").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function openConvo(id) {
  const convo = getConvo(id);
  if (!convo) return;
  if (convo.type === "chat") {
    state.activeChatId = id;
    loadChatView(convo);
    showTab("chatTab");
  } else {
    state.activeImageId = id;
    loadImageView(convo);
    showTab("imageTab");
  }
}

// ---- undo toast: a deleted chat can be brought back for 10 seconds ----
let pendingUndo = null;

function hideToast() {
  if (pendingUndo) { clearTimeout(pendingUndo.timer); pendingUndo = null; }
  toast.hidden = true;
}

function showToast(text, onUndo) {
  hideToast(); // a newer deletion replaces the older one (the older becomes permanent)
  toastText.textContent = text;
  toast.hidden = false;
  pendingUndo = { onUndo, timer: setTimeout(hideToast, 10000) };
}

toastUndo.addEventListener("click", () => {
  if (!pendingUndo) return;
  const undo = pendingUndo.onUndo;
  hideToast();
  undo();
});

/** Deletes a conversation. With `withUndo`, a non-empty one can be restored from the toast. */
function deleteConvo(id, withUndo) {
  const convo = getConvo(id);
  if (!convo) return;
  const index = state.conversations.indexOf(convo);
  const wasActiveChat = state.activeChatId === id;
  const wasActiveImage = state.activeImageId === id;

  state.conversations = state.conversations.filter((c) => c.id !== id);
  if (wasActiveChat) { state.activeChatId = null; loadChatView(null); }
  if (wasActiveImage) { state.activeImageId = null; loadImageView(null); }
  saveState();
  renderRecents();

  if (withUndo && convo.entries.length) {
    showToast(`${convo.type === "image" ? "Image chat" : "Chat"} deleted`, () => {
      state.conversations.splice(Math.min(index, state.conversations.length), 0, convo);
      // reopen it only if the user has not moved on to another conversation meanwhile
      if (wasActiveChat && state.activeChatId === null) { state.activeChatId = id; loadChatView(convo); }
      if (wasActiveImage && state.activeImageId === null) { state.activeImageId = id; loadImageView(convo); }
      saveState();
      renderRecents();
    });
  }
}

newChatBtn.addEventListener("click", () => {
  if (currentTab === "imageTab") {
    state.activeImageId = null;
    loadImageView(null);
    showTab("imageTab");
    imageQuestionInput.focus();
  } else {
    state.activeChatId = null;
    loadChatView(null);
    showTab("chatTab");
    queryInput.focus();
  }
});

let lastStatus = null;

function makeChip(text, cls, tip) {
  const s = document.createElement("span");
  s.className = "chip" + (cls ? " " + cls : "");
  s.textContent = text;
  if (tip) s.title = tip;
  return s;
}

/** Fills the status chips in every top bar. `data` = null means the backend is unreachable. */
function renderStatusChips(data) {
  const containers = Array.from(document.querySelectorAll(".sys-chips"));
  containers.forEach((c) => { c.innerHTML = ""; });
  const add = (chip) => containers.forEach((c, i) => c.appendChild(i === 0 ? chip : chip.cloneNode(true)));

  if (!data) { add(makeChip("Backend unreachable - is uvicorn running?", "bad")); return; }
  const rs = data.resource_state;
  // the RAM colour uses the same budget the router uses (free RAM + memory held by loaded models)
  const budget = data.ram_budget_gb ?? data.effective_ram_gb ?? rs.ram_available_gb;
  // same rule as the router: a model fits if it is already loaded (needs no new RAM) or its need <= budget
  const needMap = data.model_requirements_gb || {};
  const warmSet = new Set(data.loaded_models || []);
  const names = Object.keys(needMap);
  const fitting = names.filter((m) => needMap[m] <= budget || warmSet.has(m)).length;
  let ramClass = "";
  if (names.length) ramClass = fitting === names.length ? "ok" : (fitting === 0 ? "bad" : "warn");
  const loaded = data.ollama_loaded_gb || 0;
  const reasons = data.stress_reasons && data.stress_reasons.length ? ` (reduced: ${data.stress_reasons.join(", ")})` : "";
  const ramTip = `Free RAM ${rs.ram_available_gb} GB` + (loaded ? ` + ${loaded} GB held by loaded models` : "") +
    ` = ${budget} GB budget${reasons}.` + (names.length ? ` ${fitting} of ${names.length} models fit` + (warmSet.size ? ` (${Array.from(warmSet).join(", ")} already loaded).` : ".") : "");
  add(makeChip(`RAM ${rs.ram_available_gb} GB free`, ramClass, ramTip));
  add(makeChip(`CPU ${rs.cpu_percent}%`, rs.cpu_percent > 85 ? "warn" : "",
               "CPU load - above 85% the router becomes more conservative"));
  if (rs.battery_percent !== null && rs.battery_percent !== undefined) {
    const critical = rs.battery_percent < 15 && !rs.battery_charging;
    add(makeChip(rs.battery_charging ? `\u26A1 ${rs.battery_percent}%` : `\uD83D\uDD0B ${rs.battery_percent}% on battery`,
                 critical ? "bad" : "", rs.battery_charging ? "Charging" : "On battery - below 15% the router becomes more conservative"));
  }
  add(makeChip("CPU-only", "dim gpu-chip", rs.gpu_note || "CPU-only inference"));
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    lastStatus = data;
    populateModelOptions(data);
    renderStatusChips(data);
  } catch (e) {
    renderStatusChips(null);
  }
}

// ============================================================
// Chat tab - streaming; every completed exchange is saved to the active conversation
// ============================================================
let chatBusy = false;
let chatSession = 0; // bumped whenever the on-screen conversation changes (new / switch / delete)

function renderChatEntry(entry) {
  if (entry.type === "user") {
    appendMessage(chatThread, "user", entry.text);
    return;
  }
  const block = document.createElement("div");
  block.className = "answer-block";
  if (entry.routing) block.appendChild(buildRoutingPanel(entry.routing));
  const msg = document.createElement("div");
  msg.className = "message assistant";
  setRenderedHtml(msg, entry.text);
  block.appendChild(msg);
  if (entry.stopped) block.appendChild(makeStoppedTag());
  block.appendChild(makeActions(entry.text, null));
  chatThread.appendChild(block);
}

function updateChatTitle() {
  const c = chatConvo();
  chatTitle.textContent = c ? c.title : "New chat";
}

function loadChatView(convo) {
  chatSession += 1;
  chatThread.innerHTML = "";
  if (convo) convo.entries.forEach(renderChatEntry);
  updateChatTitle();
  scrollToBottom(chatScroll);
}

let chatAbort = null; // AbortController of the request in flight (the Stop button aborts it)
function stopChat() { if (chatAbort) chatAbort.abort(); }

/** While an answer is generating, the send arrow becomes a Stop button. */
function updateChatControls() {
  sendBtn.disabled = false;
  sendBtn.textContent = chatBusy ? "\u25A0" : "\u2191";
  sendBtn.title = chatBusy ? "Stop generating" : "Send";
  sendBtn.classList.toggle("stop", chatBusy);
}

function makeStoppedTag() {
  const tag = document.createElement("div");
  tag.className = "stopped-tag";
  tag.textContent = "\u25A0 Stopped - partial answer";
  return tag;
}

/** Saves a finished exchange. `activate` = the user is still looking at the conversation it came from. */
function recordChatExchange(startId, query, result, activate, stopped) {
  let convo = getConvo(startId);
  if (startId && !convo) return; // the conversation was deleted while the answer was generating
  if (!convo) {
    convo = createConvo("chat", query);
    if (activate) state.activeChatId = convo.id;
  }
  convo.entries.push(
    { type: "user", text: query },
    Object.assign({ type: "assistant", text: result.response_text, routing: slimRouting(result) },
                  stopped ? { stopped: true } : {})
  );
  convo.entries = trimEntries(convo.entries, MAX_CHAT_ENTRIES);
  convo.history.push({ role: "user", content: query }, { role: "assistant", content: result.response_text });
  convo.history = convo.history.slice(-MAX_HISTORY_MESSAGES);
  touchConvo(convo);
  saveState();
  renderRecents();
  updateChatTitle();
}

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query || chatBusy) return;

  const startConvo = chatConvo();
  const startId = startConvo ? startConvo.id : null;
  const history = startConvo ? startConvo.history.slice() : [];
  const session = chatSession;
  const live = () => session === chatSession; // is this conversation still the one on screen?
  const controller = new AbortController();
  chatAbort = controller;

  chatBusy = true;
  updateChatControls();
  const userBubble = appendMessage(chatThread, "user", query);
  queryInput.value = "";
  autosize(queryInput);

  const thinkingDiv = appendMessage(
    chatThread, "thinking",
    "Thinking... (CPU-only local inference, this can take 10-300+ seconds)"
  );

  let block = null;
  let responseDiv = null;
  let fullText = "";
  let finalResult = null;
  let errorShown = false;
  let routingData = null;

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode: modeSelect.value, history, model: manualModel() }),
      signal: controller.signal,
    });

    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({}));
      thinkingDiv.remove();
      appendMessage(chatThread, "error", `Error: ${err.detail || res.statusText}`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete trailing chunk for next read

      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const item = JSON.parse(part.slice(6));

        if (item.event === "routing") {
          thinkingDiv.remove();
          routingData = item.data;
          block = document.createElement("div");
          block.className = "answer-block";
          chatThread.appendChild(block);
          block.appendChild(buildRoutingPanel(item.data));
          responseDiv = appendMessage(block, "assistant", "");
        } else if (item.event === "token") {
          const stick = isNearBottom(chatScroll);
          fullText += item.text;
          responseDiv.textContent = fullText; // plain text while streaming - markdown applied at the end
          if (stick && live()) scrollToBottom(chatScroll);
        } else if (item.event === "done") {
          finalResult = item.data;
          setRenderedHtml(responseDiv, finalResult.response_text);
          // refresh the panel summary now that final timing is known
          const summary = block.querySelector("summary");
          summary.textContent =
            `${finalResult.category} -> ${finalResult.selected_model}${routingTag(finalResult)} ` +
            `(${finalResult.actual_response_time_sec}s)`;
          block.appendChild(makeActions(finalResult.response_text, null));
          if (live()) scrollToBottom(chatScroll);
        } else if (item.event === "error") {
          thinkingDiv.remove();
          errorShown = true;
          if (block && responseDiv && !responseDiv.textContent.trim()) block.remove(); // nothing arrived
          appendMessage(chatThread, "error", `Error: ${item.message}`);
        }
      }
    }

    // Stream closed without a "done" or "error" event (e.g. connection dropped).
    if (!finalResult && !errorShown) {
      thinkingDiv.remove();
      if (block && responseDiv && !responseDiv.textContent.trim()) block.remove();
      appendMessage(chatThread, "error",
        "The response ended unexpectedly (no completion signal). Check that uvicorn and Ollama are still running.");
    }

    // Only COMPLETED exchanges are saved, so a refresh never shows a dangling question.
    if (finalResult) recordChatExchange(startId, query, finalResult, live());
  } catch (e) {
    thinkingDiv.remove();
    if (e && e.name === "AbortError") {
      // the user pressed Stop
      if (fullText.trim() && block && routingData) {
        // keep what was generated so far, marked as stopped, and save it like a normal answer
        setRenderedHtml(responseDiv, fullText);
        const summary = block.querySelector("summary");
        if (summary) {
          summary.textContent = `${routingData.category} -> ${routingData.selected_model}${routingTag(routingData)} (stopped)`;
        }
        block.appendChild(makeStoppedTag());
        block.appendChild(makeActions(fullText, null));
        recordChatExchange(startId, query, Object.assign({}, routingData, { response_text: fullText }), live(), true);
      } else {
        // stopped before any text arrived: take the question back so it can be edited and resent
        if (block) block.remove();
        userBubble.remove();
        if (!queryInput.value) { queryInput.value = query; autosize(queryInput); }
      }
    } else {
      appendMessage(chatThread, "error", describeRequestError(e));
    }
  } finally {
    chatAbort = null;
    chatBusy = false;
    updateChatControls();
    refreshStatus();
  }
}

sendBtn.addEventListener("click", () => (chatBusy ? stopChat() : sendQuery()));
queryInput.addEventListener("input", () => autosize(queryInput));
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!chatBusy) sendQuery();
  }
});
modeSelect.addEventListener("change", () => { state.mode = modeSelect.value; saveState(); });

// ---- manual model choice: "Auto (router)" or one specific model for every message ----
const DEFAULT_CHAT_NOTE = chatNote.textContent;

/** The model the user forced, or null when the router should decide. */
function manualModel() {
  return state.model && state.model !== "auto" && !modelSelect.disabled ? state.model : null;
}

function ensureModelOption(name) {
  if (!Array.from(modelSelect.options).some((o) => o.value === name)) {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    modelSelect.appendChild(o);
  }
}

function populateModelOptions(data) {
  const models = Array.isArray(data.live_models) ? data.live_models : [];
  const wanted = ["auto"].concat(models);
  const current = Array.from(modelSelect.options).map((o) => o.value);
  if (wanted.join("|") !== current.join("|")) {
    modelSelect.innerHTML = "";
    wanted.forEach((m) => {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = m === "auto" ? "Auto (router)" : m;
      modelSelect.appendChild(o);
    });
  }
  if (!wanted.includes(state.model)) { state.model = "auto"; saveState(); } // model no longer available
  modelSelect.value = state.model;
  const supported = data.manual_model_supported !== false;
  modelSelect.disabled = !supported;
  modelSelect.title = supported ? "" : "Manual model choice needs the router patch (patch_router_manual.py)";
  applyModelUi();
}

function applyModelUi() {
  const manual = manualModel();
  modeSelect.disabled = !!manual; // the scoring mode does not matter when the model is fixed
  updateModelHint();
}

function updateModelHint() {
  const m = manualModel();
  chatNote.textContent = "";
  chatNote.classList.toggle("manual", !!m);
  if (!m) { chatNote.textContent = DEFAULT_CHAT_NOTE; return; }
  chatNote.appendChild(document.createTextNode(
    `Manual: every message goes to ${m}. The router still shows what it would have chosen.`));
  const need = lastStatus && lastStatus.model_requirements_gb ? lastStatus.model_requirements_gb[m] : null;
  // same RAM figure the router uses: free RAM + memory Ollama would unload (see /api/status)
  const avail = lastStatus
    ? (lastStatus.ram_budget_gb ?? lastStatus.effective_ram_gb ?? (lastStatus.resource_state && lastStatus.resource_state.ram_available_gb))
    : null;
  const alreadyLoaded = !!(lastStatus && Array.isArray(lastStatus.loaded_models) && lastStatus.loaded_models.includes(m));
  if (alreadyLoaded) {
    const span = document.createElement("span");
    span.className = "note-ok";
    span.textContent = " Already loaded in memory - no extra RAM needed.";
    chatNote.appendChild(span);
  } else if (need && avail !== null && avail !== undefined) {
    const span = document.createElement("span");
    if (avail >= need) {
      span.className = "note-ok";
      span.textContent = ` Needs ~${need} GB, ${avail} GB available.`;
    } else {
      span.className = "note-warn";
      span.textContent = ` Warning: needs ~${need} GB but only ${avail} GB is available ` +
                         `(free RAM plus memory Ollama would unload) - it may be slow.`;
    }
    chatNote.appendChild(span);
  }
}

modelSelect.addEventListener("change", () => { state.model = modelSelect.value; saveState(); applyModelUi(); });
clearChatBtn.addEventListener("click", () => {
  const c = chatConvo();
  if (!c) { loadChatView(null); return; }
  deleteConvo(c.id, true);
});
wireScrollDown(chatScroll, chatScrollDown);

// ============================================================
// Image tab - attach once, keep asking. The image is uploaded to the server a single time
// and referenced by id afterwards, so follow-up questions never re-upload or re-select it.
// ============================================================
const IMAGE_CONTEXT_EXCHANGES = 2;   // earlier Q&A pairs sent along so "explain the second point" works
const IMAGE_CONTEXT_MAX_CHARS = 1200;
const THUMB_MAX_PX = 160;

let imageBusy = false;
let compareMode = false;             // the Compare toggle: ON = the next send runs BOTH vision models
let imageAbort = null;                // AbortController of the ask/compare in flight
let imageSession = 0;                 // bumped whenever the on-screen image conversation changes
const entryEls = new WeakMap();       // entry -> its DOM element (kept out of the saved data)

function currentImage() { const c = imgConvo(); return c ? c.currentImage : null; }

function normalizeQuestion(q) {
  return String(q).toLowerCase().replace(/\s+/g, " ").replace(/[?.!\s]+$/, "").trim();
}

function makeThumbnail(file) {
  return new Promise((resolve) => {
    if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function") { resolve(""); return; }
    const url = URL.createObjectURL(file);
    const img = new Image();
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      try { URL.revokeObjectURL(url); } catch (e) { /* ignore */ }
      resolve(value);
    };
    const timer = setTimeout(() => finish(""), 4000);
    img.onload = () => {
      clearTimeout(timer);
      try {
        const scale = Math.min(1, THUMB_MAX_PX / Math.max(img.width, img.height, 1));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#fff"; // JPEG has no transparency
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        finish(canvas.toDataURL("image/jpeg", 0.7));
      } catch (e) { finish(""); }
    };
    img.onerror = () => { clearTimeout(timer); finish(""); };
    img.src = url;
  });
}

/** Earlier Q&A about THIS image only (a different image starts with a clean context). */
function buildImageContext(convo, imageId, beforeIndex) {
  const relevant = convo.entries.slice(0, beforeIndex)
    .filter((e) => e.imageId === imageId && (e.type === "user" || e.type === "assistant"));
  const pairs = [];
  for (let i = 0; i < relevant.length - 1; i++) {
    if (relevant[i].type === "user" && relevant[i + 1].type === "assistant") {
      pairs.push([relevant[i].text, relevant[i + 1].text]);
      i += 1;
    }
  }
  return pairs.slice(-IMAGE_CONTEXT_EXCHANGES).flatMap(([q, a]) => [
    { role: "user", content: q },
    { role: "assistant", content: a.slice(0, IMAGE_CONTEXT_MAX_CHARS) },
  ]);
}

function showImageNote(text) {
  const n = appendMessage(imageThread, "note", text);
  setTimeout(() => n.remove(), 12000);
}

function revealEntry(entry) {
  const el = entryEls.get(entry);
  if (!el) return;
  if (el.scrollIntoView) el.scrollIntoView({ block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1700);
}

function updateImageTitle() {
  const c = imgConvo();
  imageTitle.textContent = c ? c.title : "New image chat";
}

function updateImageComposer() {
  const img = currentImage();
  imageChip.hidden = !img;
  if (img) {
    imageChipThumb.src = img.thumb || "";
    imageChipThumb.style.display = img.thumb ? "" : "none";
    imageChipName.textContent = img.name;
  }
  const stoppable = imageBusy && imageAbort !== null;
  imageSendBtn.textContent = stoppable ? "\u25A0" : "\u2191";
  imageSendBtn.title = stoppable ? "Stop waiting (the server may finish this answer in the background)"
                                 : (compareMode ? "Run the comparison" : "Ask");
  imageSendBtn.classList.toggle("stop", stoppable);
  imageSendBtn.disabled = stoppable ? false : (imageBusy || !img);
  if (!img) compareMode = false;       // nothing to compare without an image
  imageCompareBtn.disabled = imageBusy || !img;
  imageCompareBtn.classList.toggle("active", compareMode);
  imageCompareBtn.setAttribute("aria-pressed", compareMode ? "true" : "false");
  imageCompareBtn.title = compareMode
    ? "Compare mode is ON - the next send runs both vision models. Click to turn it off."
    : "Turn on, then press Enter or the arrow to run both vision models on your question";
  attachBtn.disabled = imageBusy;
  imageQuestionInput.placeholder = !img
    ? "Attach an image with + (or paste / drag & drop), then ask about it"
    : (compareMode
        ? "Compare mode: BOTH vision models will answer this question  (Enter to run)"
        : "Ask about this image...  (Enter to send, Shift+Enter for newline)");
  document.querySelectorAll("#imageThread .rerun-btn").forEach((b) => { b.disabled = imageBusy; });
}

function setImageBusy(busy) { imageBusy = busy; updateImageComposer(); }

/** Why a vision model produced no visible text (the router reports done_reason / thinking_chars when it can). */
function emptyReason(r) {
  if (r.done_reason === "length") {
    return "it used its whole token limit before writing an answer" +
           (r.thinking_chars ? ", spending it on hidden 'thinking'" : "");
  }
  if (r.thinking_chars) return "it produced only hidden 'thinking', no answer";
  return "this model returned nothing";
}

function renderImageEntry(e) {
  let el;
  if (e.type === "image") {
    el = document.createElement("div");
    el.className = "message user image-msg";
    if (e.thumb) {
      const img = document.createElement("img");
      img.src = e.thumb; img.alt = e.name; img.className = "msg-thumb";
      el.appendChild(img);
    }
    const cap = document.createElement("span");
    cap.className = "image-cap"; cap.textContent = e.name;
    el.appendChild(cap);
    imageThread.appendChild(el);
  } else if (e.type === "user") {
    el = appendMessage(imageThread, "user", e.text);
  } else if (e.type === "assistant") {
    el = document.createElement("div");
    el.className = "answer-block";
    if (e.rerun) {
      const tag = document.createElement("div");
      tag.className = "rerun-tag"; tag.textContent = "\u21BB Re-run of an earlier answer";
      el.appendChild(tag);
    }
    if (e.routing) el.appendChild(buildRoutingPanel(e.routing));
    const msg = document.createElement("div");
    msg.className = "message assistant";
    setRenderedHtml(msg, e.text);
    el.appendChild(msg);
    el.appendChild(makeActions(e.text, () => askAboutImage(e.question, e)));
    imageThread.appendChild(el);
  } else if (e.type === "compare") {
    el = document.createElement("div");
    el.className = "answer-block";
    e.results.forEach((r) => {
      const div = document.createElement("div");
      div.className = "compare-result";
      const note = r.constrained ? " (ran under RAM constraint)" : "";
      const body = (r.text || "").trim()
        ? renderMarkdown(r.text)
        : `<p><em>(empty response - ${emptyReason(r)})</em></p>`;
      div.innerHTML = `<h4>${r.model} - ${r.seconds}s${note}</h4>` + body;
      el.appendChild(div);
      renderMathIn(div);
    });
    imageThread.appendChild(el);
  }
  if (el) entryEls.set(e, el);
  scrollToBottom(imageScroll);
  updateImageComposer();
}

function loadImageView(convo) {
  imageSession += 1;
  imageThread.innerHTML = "";
  if (convo) convo.entries.forEach(renderImageEntry);
  updateImageTitle();
  updateImageComposer();
  scrollToBottom(imageScroll);
}

function saveImageConvo(convo) {
  convo.entries = trimEntries(convo.entries, MAX_IMAGE_ENTRIES);
  touchConvo(convo);
  saveState();
  renderRecents();
  updateImageTitle();
}

async function attachImage(file) {
  if (!file || imageBusy) return;
  if (!file.type || !file.type.startsWith("image/")) { showImageNote("That file is not an image."); return; }

  const session = imageSession;
  setImageBusy(true);
  const status = appendMessage(imageThread, "thinking", "Uploading image...");
  try {
    const thumb = await makeThumbnail(file);
    const form = new FormData();
    form.append("image", file);
    const res = await fetch("/api/image/upload", { method: "POST", body: form });
    status.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendMessage(imageThread, "error", `Error: ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    const name = file.name || "pasted image";
    let convo = imgConvo();
    if (!convo) {
      convo = createConvo("image", `Image: ${name}`);
      state.activeImageId = convo.id;
    }
    convo.currentImage = { imageId: data.image_id, name, thumb };
    const entry = { type: "image", imageId: data.image_id, name, thumb };
    convo.entries.push(entry);
    saveImageConvo(convo);
    if (session === imageSession) renderImageEntry(entry);
  } catch (e) {
    status.remove();
    appendMessage(imageThread, "error", describeRequestError(e));
  } finally {
    setImageBusy(false);
  }
}

async function askAboutImage(question, rerunOf) {
  const convo = imgConvo();
  if (!convo) { showImageNote("Attach an image first."); return; }
  const imageId = rerunOf ? rerunOf.imageId : (convo.currentImage && convo.currentImage.imageId);
  if (!imageId) { showImageNote("Attach an image first (+ button, paste, or drag & drop)."); return; }
  if (!question || imageBusy) return;

  // Same image + same question already answered: show that answer instead of re-running the model.
  if (!rerunOf) {
    const wanted = normalizeQuestion(question);
    const existing = convo.entries.slice().reverse().find((e) =>
      e.type === "assistant" && e.imageId === imageId && normalizeQuestion(e.question) === wanted);
    if (existing) {
      revealEntry(existing);
      showImageNote("Already answered above for this image - not re-running. Use \"\u21BB Re-run\" under that answer for a fresh one.");
      return;
    }
  }

  const session = imageSession;
  const convoId = convo.id;
  const context = buildImageContext(convo, imageId, rerunOf ? convo.entries.indexOf(rerunOf) : convo.entries.length);
  let userEntry = null;
  if (rerunOf) {
    showImageNote(`\u21BB Re-running: ${truncate(question, 60)}`);
  } else {
    userEntry = { type: "user", imageId, text: question };
    renderImageEntry(userEntry); // shown immediately, saved once the answer arrives
    imageQuestionInput.value = "";
    autosize(imageQuestionInput);
  }

  const controller = new AbortController();
  imageAbort = controller;
  setImageBusy(true);
  const thinking = appendMessage(imageThread, "thinking", "Analyzing image...");
  try {
    const res = await fetch("/api/image/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId, question, history: context }),
      signal: controller.signal,
    });
    thinking.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      if (res.status === 404) {
        const c = getConvo(convoId);
        if (c && c.currentImage && c.currentImage.imageId === imageId) { c.currentImage = null; saveState(); }
        appendMessage(imageThread, "error", "This image is no longer available on the server - please attach it again.");
      } else {
        appendMessage(imageThread, "error", `Error: ${err.detail || res.statusText}`);
      }
      if (userEntry && !imageQuestionInput.value) imageQuestionInput.value = question; // let the user retry
      return;
    }
    const result = await res.json();
    const target = getConvo(convoId);
    if (!target) return; // deleted while the answer was generating
    const answer = { type: "assistant", imageId, question, text: result.response_text,
                     routing: slimRouting(result), rerun: !!rerunOf };
    if (userEntry) target.entries.push(userEntry);
    target.entries.push(answer);
    saveImageConvo(target);
    if (session === imageSession) renderImageEntry(answer);
  } catch (e) {
    thinking.remove();
    if (e && e.name === "AbortError") {
      if (userEntry) { // take the question back so it can be edited and resent
        const el = entryEls.get(userEntry);
        if (el) el.remove();
        if (!imageQuestionInput.value) { imageQuestionInput.value = question; autosize(imageQuestionInput); }
      }
      showImageNote("Stopped. The server may still be finishing that image for a moment - wait a little before asking again.");
    } else {
      appendMessage(imageThread, "error", describeRequestError(e));
    }
  } finally {
    imageAbort = null;
    setImageBusy(false);
  }
}

async function compareImageModels() {
  const convo = imgConvo();
  const img = convo && convo.currentImage;
  const question = imageQuestionInput.value.trim();
  if (!img) { showImageNote("Attach an image first."); return; }
  if (!question) { showImageNote("Type a question to compare the two models on."); return; }
  if (imageBusy) return;
  compareMode = false; // one-shot: a comparison takes twice as long, so the toggle does not stay on by itself

  const session = imageSession;
  const convoId = convo.id;
  const userEntry = { type: "user", imageId: img.imageId, text: `[compare] ${question}` };
  renderImageEntry(userEntry);
  imageQuestionInput.value = "";
  autosize(imageQuestionInput);

  const controller = new AbortController();
  imageAbort = controller;
  setImageBusy(true);
  const thinking = appendMessage(imageThread, "thinking",
    "Running both vision models for comparison (this takes longer)...");
  try {
    const res = await fetch("/api/image/compare_saved", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: img.imageId, question }),
      signal: controller.signal,
    });
    thinking.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendMessage(imageThread, "error", `Error: ${err.detail || res.statusText}`);
      return;
    }
    const result = await res.json();
    const target = getConvo(convoId);
    if (!target) return;
    const entry = {
      type: "compare", imageId: img.imageId, question,
      results: Object.entries(result.results_by_model).map(([model, r]) => ({
        model, seconds: r.actual_response_time_sec, constrained: !!r.ran_under_ram_constraint, text: r.response_text || "",
        done_reason: r.done_reason || null, thinking_chars: r.thinking_chars || 0,
      })),
    };
    target.entries.push(userEntry, entry);
    saveImageConvo(target);
    if (session === imageSession) renderImageEntry(entry);
  } catch (e) {
    thinking.remove();
    if (e && e.name === "AbortError") {
      const el = entryEls.get(userEntry);
      if (el) el.remove();
      if (!imageQuestionInput.value) { imageQuestionInput.value = question; autosize(imageQuestionInput); }
      showImageNote("Stopped. The server may still be finishing the comparison - wait a little before asking again.");
    } else {
      appendMessage(imageThread, "error", describeRequestError(e));
    }
  } finally {
    imageAbort = null;
    setImageBusy(false);
  }
}

function submitImageQuestion() {
  const q = imageQuestionInput.value.trim();
  if (!currentImage()) { showImageNote("Attach an image first (+ button, paste, or drag & drop)."); return; }
  if (!q) { showImageNote(compareMode ? "Type a question to compare the two models on." : "Type a question about the image."); return; }
  if (compareMode) compareImageModels();
  else askAboutImage(q, null);
}

attachBtn.addEventListener("click", () => imageFileInput.click());
imageFileInput.addEventListener("change", () => {
  const file = imageFileInput.files[0];
  imageFileInput.value = ""; // allows picking the same file again later
  attachImage(file);
});
imageSendBtn.addEventListener("click", () => (imageAbort ? imageAbort.abort() : submitImageQuestion()));
// Compare is a TOGGLE (it does not run anything by itself): turn it on, type the question, press Enter or the arrow.
imageCompareBtn.addEventListener("click", () => {
  if (imageCompareBtn.disabled) return;
  compareMode = !compareMode;
  updateImageComposer();
  imageQuestionInput.focus();
});
imageQuestionInput.addEventListener("input", () => autosize(imageQuestionInput));
imageQuestionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!imageBusy) submitImageQuestion();
  }
});
imageQuestionInput.addEventListener("paste", (e) => {
  const files = e.clipboardData && e.clipboardData.files ? Array.from(e.clipboardData.files) : [];
  const pic = files.find((f) => f.type && f.type.startsWith("image/"));
  if (pic) { e.preventDefault(); attachImage(pic); }
});
const imageTabPanel = $("imageTab");
imageTabPanel.addEventListener("dragover", (e) => e.preventDefault());
imageTabPanel.addEventListener("drop", (e) => {
  e.preventDefault();
  const files = e.dataTransfer && e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
  const pic = files.find((f) => f.type && f.type.startsWith("image/"));
  if (pic) attachImage(pic);
});

clearImageBtn.addEventListener("click", () => {
  const c = imgConvo();
  if (c) { c.currentImage = null; saveState(); }
  updateImageComposer();
});
imageClearAllBtn.addEventListener("click", () => {
  const c = imgConvo();
  if (!c) { loadImageView(null); return; }
  deleteConvo(c.id, true);
});
wireScrollDown(imageScroll, imageScrollDown);

// ============================================================
// History tab
// ============================================================
const historyTableBody = document.getElementById("historyTableBody");
const historySummary = document.getElementById("historySummary");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

/** Builds one <td> using textContent (never innerHTML - queries are user-typed text). */
function makeCell(text, className, title) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  if (title) td.title = title; // hover shows the full text of truncated cells
  return td;
}

function isTruthyFlag(v) {
  return v === true || String(v).toLowerCase() === "true";
}

function formatTimestamp(ts) {
  // log stores ISO strings like 2026-10-12T16:34:53 - show as "2026-10-12 16:34"
  return String(ts || "").replace("T", " ").slice(0, 16);
}

async function loadHistory() {
  refreshHistoryBtn.disabled = true;
  historySummary.textContent = "Loading...";
  try {
    const res = await fetch("/api/history?limit=50");
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      historyTableBody.innerHTML = "";
      historySummary.textContent = `Error: ${err.detail || res.statusText}`;
      return;
    }
    const data = await res.json();
    historyTableBody.innerHTML = "";

    if (data.rows.length === 0) {
      historySummary.textContent = "No interactions logged yet.";
      return;
    }

    for (const r of data.rows) {
      const tr = document.createElement("tr");
      const source = r.source || "-";
      const switched = isTruthyFlag(r.model_switch);
      const secs = r.actual_response_time_sec;

      tr.appendChild(makeCell(formatTimestamp(r.timestamp)));
      tr.appendChild(makeCell(source, `source-${source}`));
      tr.appendChild(makeCell(r.query, "", r.query));
      tr.appendChild(makeCell(r.category || "-"));
      tr.appendChild(makeCell(r.selected_model || "-"));
      tr.appendChild(makeCell(switched ? "yes" : "", switched ? "switch-yes" : ""));
      tr.appendChild(makeCell(secs === "" || secs === undefined ? "-" : secs));
      historyTableBody.appendChild(tr);
    }
    historySummary.textContent =
      `Showing latest ${data.rows.length} of ${data.total_logged} logged interactions`;
  } catch (e) {
    historyTableBody.innerHTML = "";
    historySummary.textContent = describeRequestError(e);
  } finally {
    refreshHistoryBtn.disabled = false;
  }
}

refreshHistoryBtn.addEventListener("click", loadHistory);

// ============================================================
// Startup
// ============================================================
loadState();
if (state.mode && Array.from(modeSelect.options).some((o) => o.value === state.mode)) modeSelect.value = state.mode;
document.body.classList.toggle("sidebar-collapsed", state.sidebarCollapsed === null ? isNarrow() : !!state.sidebarCollapsed);
if (state.model && state.model !== "auto") { ensureModelOption(state.model); modelSelect.value = state.model; }
applyModelUi();
loadChatView(chatConvo());
loadImageView(imgConvo());
refreshStatus();
setInterval(refreshStatus, 10000);
showTab(["chatTab", "imageTab", "historyTab"].includes(state.lastTab) ? state.lastTab : "chatTab");