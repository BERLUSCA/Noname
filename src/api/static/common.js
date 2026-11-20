// Shared helpers for the page. Plain JavaScript, no framework.
// The player name is stored only when the user confirms it on the home page ("Entra").

const PLAYER_KEY = "quantone.player";

function getPlayer() {
  return localStorage.getItem(PLAYER_KEY) || "";
}

function setPlayer(name) {
  localStorage.setItem(PLAYER_KEY, name);
}

function clearPlayer() {
  localStorage.removeItem(PLAYER_KEY);
}

function requirePlayer() {
  const name = getPlayer();
  if (!name) throw new Error("Prima entra con il tuo nome nella pagina Home");
  return name;
}

async function api(method, path, body, extraHeaders) {
  const headers = { ...(extraHeaders || {}) };
  if (body) headers["Content-Type"] = "application/json";
  const res = await fetch(path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Errore ${res.status}`);
  return data;
}

// Built with textContent only: server text is never interpreted as HTML.
function el(tag, text, cls) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

// A ticket stub: a short label on the left, the value (score, challenge code) on the right.
function ticket(label, value, cls) {
  const box = el("div", "", cls ? `ticket ${cls}` : "ticket");
  box.appendChild(el("span", label, "stub"));
  box.appendChild(el("span", String(value), "main"));
  return box;
}

// Marks each question as right / wrong / skipped after scoring. The server sends only booleans, never the correct option.
function markResults(prefix, questions, results) {
  questions.forEach((q, i) => {
    const box = document.querySelector(`input[name="${prefix}${q.position}"]`)?.closest(".question");
    if (!box) return;
    const answered = document.querySelector(`input[name="${prefix}${q.position}"]:checked`);
    box.classList.remove("right", "wrong", "skipped");
    box.classList.add(!answered ? "skipped" : results[i] ? "right" : "wrong");
  });
}

function showError(container, err) {
  container.replaceChildren(el("p", err.message, "bad"));
}

// Radio buttons for a list of questions (from the server: no correct answers are included).
function questionForm(prefix, questions) {
  const form = document.createElement("div");
  questions.forEach((q) => {
    const box = el("div", "", "question");
    box.appendChild(el("strong", `${q.position + 1}. ${q.question}`));
    q.options.forEach((opt, i) => {
      const label = el("label", "");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = `${prefix}${q.position}`;
      radio.value = String(i);
      label.appendChild(radio);
      label.appendChild(document.createTextNode(` ${opt}`));
      box.appendChild(label);
    });
    form.appendChild(box);
  });
  return form;
}

function collectAnswers(prefix, questions) {
  return questions.map((q) => {
    const checked = document.querySelector(`input[name="${prefix}${q.position}"]:checked`);
    return checked ? Number(checked.value) : null;
  });
}

// Header status line: the current player and the state of the models. Safe to call again after login or logout.
function renderHeader() {
  const who = document.getElementById("who");
  if (!who) return;
  const line = document.createElement("span");
  line.id = "warm";
  who.textContent = getPlayer() ? `Giocatore: ${getPlayer()}` : "Non hai ancora effettuato l'accesso";
  who.append("  ·  ", line);
  startWarmUp();
}

// Loads the models as soon as the page opens (POST /warmup), so the first question is not slow.
// The status is shown in the header; it never blocks the page.
async function startWarmUp() {
  const box = document.getElementById("warm");
  if (!box) return;
  box.textContent = "Modelli: caricamento…";
  box.className = "muted";
  try {
    const data = await api("POST", "/warmup");
    if (data.ready) {
      box.textContent = "Modelli: pronti";
      box.className = "ok";
    } else {
      const problems = Object.entries(data.components)
        .filter(([, v]) => v !== "ok")
        .map(([k, v]) => `${k}: ${v}`);
      box.textContent = `Modelli non pronti: ${problems.join(" · ")}`;
      box.className = "bad";
    }
  } catch (err) {
    box.textContent = `Modelli: ${err.message}`;
    box.className = "bad";
  }
}
