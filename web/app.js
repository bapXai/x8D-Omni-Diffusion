"use strict";

/* x8D byte-native chat frontend — talks to the OpenAI-compatible endpoint.
 * Usage is counted in BYTES (byte law), never tokens.
 */

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const promptEl = $("prompt");
const sendBtn = $("send");
const historyEl = $("history");
const syslineEl = $("sysline");
const ramPill = $("ram-pill");

const ENDPOINT = "/v1/chat/completions";
const HISTORY_KEY = "x8d.chat.history";

const history = []; // {role, content, meta}
let streaming = false;

/* ---------------- render ---------------- */

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function addMessage(role, content, meta) {
  const welcome = messagesEl.querySelector(".welcome");
  if (welcome) welcome.remove();

  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🧑" : "⬡";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    wrap.appendChild(m);
  }
  messagesEl.appendChild(wrap);
  scrollBottom();
  return bubble;
}

/* ---------------- history ---------------- */

function saveHistory() {
  const items = history
    .filter((h) => h.role === "user")
    .map((h) => h.content.slice(0, 60));
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
  renderHistory();
}

function renderHistory() {
  let items = [];
  try {
    items = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch (e) { items = []; }
  historyEl.innerHTML = "";
  items.reverse().forEach((text, i) => {
    const div = document.createElement("div");
    div.className = "hitem";
    div.textContent = text;
    div.addEventListener("click", () => {
      promptEl.value = text;
      promptEl.focus();
    });
    historyEl.appendChild(div);
  });
}

/* ---------------- telemetry ---------------- */

async function refreshTelemetry() {
  try {
    const res = await fetch("/telemetry");
    const data = await res.json();
    ramPill.textContent = `RSS ${data.rss_mb} MB · blk ${data.blocks} · io ${data.io_mb} MB`;
    syslineEl.textContent =
      `label: ${data.label}\n` +
      `io: ${data.io_mb} MB · fault: ${data.fault_mb} MB\n` +
      `blocks: ${data.blocks} · hit pin/lru: ${data.hits_pin}/${data.hits_lru}\n` +
      `elapsed: ${data.elapsed_s}s\n` +
      `mode: ${data.mode}`;
  } catch (e) {
    /* server telemetry optional */
  }
}

/* ---------------- streaming send ---------------- */

async function sendMessage() {
  const text = promptEl.value.trim();
  if (!text || streaming) return;

  history.push({ role: "user", content: text });
  addMessage("user", text);
  promptEl.value = "";
  promptEl.style.height = "auto";
  saveHistory();

  streaming = true;
  sendBtn.disabled = true;
  const bubble = addMessage("assistant", "");
  bubble.parentElement.classList.add("streaming");

  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "x8d-byte-diffusion",
        messages: history.filter((m) => m.role === "user" || m.role === "assistant"),
        stream: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      bubble.textContent = `error: ${(err.error && err.error.message) || res.status}`;
      bubble.parentElement.classList.remove("streaming");
      return;
    }

    if (!res.body) {
      bubble.textContent = "streaming body unavailable — use non-stream mode";
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let acc = "";
    let meta = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const obj = JSON.parse(payload);
          const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
          if (delta && typeof delta.content === "string") {
            acc += delta.content;
            bubble.textContent = acc;
            scrollBottom();
          }
          if (obj.usage) {
            meta = `bytes — prompt ${obj.usage.prompt_tokens} · completion ${obj.usage.completion_tokens} · total ${obj.usage.total_tokens}`;
          }
        } catch (e) { /* partial line */ }
      }
    }
    bubble.parentElement.classList.remove("streaming");
    if (meta) {
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = meta;
      bubble.parentElement.appendChild(m);
    }
    history.push({ role: "assistant", content: acc });
  } catch (e) {
    bubble.textContent = `network error: ${e.message}`;
    bubble.parentElement.classList.remove("streaming");
  } finally {
    streaming = false;
    sendBtn.disabled = false;
    refreshTelemetry();
  }
}

/* ---------------- wiring ---------------- */

function autoGrow() {
  promptEl.style.height = "auto";
  promptEl.style.height = Math.min(promptEl.scrollHeight, 180) + "px";
}

sendBtn.addEventListener("click", sendMessage);
promptEl.addEventListener("input", autoGrow);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendMessage();
  }
});
$("new-chat").addEventListener("click", () => {
  history.length = 0;
  messagesEl.innerHTML =
    `<div class="welcome">
       <h1>Byte-native diffusion chat</h1>
       <p>256 byte states + 8 specials. No tokens. No tokenizer.</p>
       <p>Every message is encoded to UTF-8 bytes, masked onto a diffusion canvas, denoised, and decoded back.</p>
     </div>`;
  saveHistory();
});

renderHistory();
refreshTelemetry();
setInterval(refreshTelemetry, 5000);
promptEl.focus();
