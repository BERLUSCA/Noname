"""Conversation memory: a sliding window of recent messages, rendered as DATA for the model.

Why: sending the whole history costs more tokens on every turn. We send the last WINDOW_SIZE messages.
Stored messages are user text: they cannot close the data block (guardrail 4).
"""

from src.quiz.generator import _as_data

WINDOW_SIZE = 6  # last 6 messages = 3 exchanges
MAX_MESSAGE_CHARS = 1000
HISTORY_START = "<<<STORIA"
HISTORY_END = "STORIA>>>"
ROLE_LABELS = {"user": "UTENTE", "assistant": "ASSISTENTE"}


def _safe(text: str) -> str:
    return _as_data(text).replace(HISTORY_START, "").replace(HISTORY_END, "")


def render_history(messages: list[dict], session_quiz_ids: list[str]) -> str:
    """Renders the window and the session's quiz ids as one data block. Empty string if nothing to show."""
    if not messages and not session_quiz_ids:
        return ""
    lines = []
    if session_quiz_ids:
        lines.append("Quiz creati in questa sessione (id): " + ", ".join(_safe(i) for i in session_quiz_ids))
    for m in messages:
        label = ROLE_LABELS.get(m["role"], "UTENTE")
        lines.append(f"{label}: {_safe(m['content'])[:MAX_MESSAGE_CHARS]}")
    return f"{HISTORY_START}\n" + "\n".join(lines) + f"\n{HISTORY_END}"
