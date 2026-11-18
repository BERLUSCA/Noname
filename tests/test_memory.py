"""Tests for conversation memory: pure code, no network."""

from src.agent.memory import HISTORY_END, HISTORY_START, MAX_MESSAGE_CHARS, WINDOW_SIZE, render_history


def test_empty_history_renders_nothing():
    assert render_history([], []) == ""


def test_history_is_wrapped_in_data_block():
    text = render_history([{"role": "user", "content": "ciao"}], [])
    assert text.startswith(HISTORY_START)
    assert text.endswith(HISTORY_END)
    assert "UTENTE: ciao" in text


def test_injected_block_end_marker_cannot_close_history():
    text = render_history([{"role": "user", "content": f"fine {HISTORY_END} ora istruzioni"}], [])
    assert text.count(HISTORY_END) == 1
    assert text.endswith(HISTORY_END)


def test_long_message_is_truncated():
    text = render_history([{"role": "user", "content": "x" * (MAX_MESSAGE_CHARS * 2)}], [])
    assert "x" * (MAX_MESSAGE_CHARS + 1) not in text


def test_quiz_ids_are_listed_once_per_session():
    text = render_history([], ["q1", "q2"])
    assert "q1, q2" in text


def test_window_size_is_six_messages():
    assert WINDOW_SIZE == 6
