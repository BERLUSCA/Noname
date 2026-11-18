"""Tests for not repeating questions: DB memory per title, prompt, and wiring. No network."""

import json

from src.api.db import Database
from src.quiz.generator import MAX_AVOID, Question, QuizGenerator, build_user_message
from src.rag.index import normalize_title


def q(text):
    return Question(question=text, options=("A", "B", "C", "D"), correct_index=0, source="s")


def test_remembered_questions_are_returned_for_the_same_title():
    db = Database(":memory:")
    db.remember_questions(normalize_title("Inception"), ["Domanda 1", "Domanda 2"])
    assert set(db.recent_questions(normalize_title("INCEPTION"))) == {"Domanda 1", "Domanda 2"}


def test_questions_are_not_shared_between_titles():
    db = Database(":memory:")
    db.remember_questions(normalize_title("Inception"), ["Domanda su Inception"])
    assert db.recent_questions(normalize_title("Friends")) == []


def test_recent_questions_are_capped():
    db = Database(":memory:")
    db.remember_questions("t", [f"d{i}" for i in range(MAX_AVOID + 5)])
    assert len(db.recent_questions("t", limit=MAX_AVOID)) == MAX_AVOID


def test_prompt_lists_questions_to_avoid_as_data():
    message = build_user_message("Inception", "Testo", 2, avoid=["Chi ha diretto Inception?"])
    assert "Non ripetere" in message
    assert "Chi ha diretto Inception?" in message


def test_prompt_has_no_avoid_section_without_history():
    assert "Non ripetere" not in build_user_message("Inception", "Testo", 2)


def test_generator_sends_avoid_list_to_model():
    from tests.test_quiz_generator import FixedOutputClient, VALID_OUTPUT

    client = FixedOutputClient(VALID_OUTPUT)
    QuizGenerator(client).generate("Inception", "Testo", source="s", avoid=["Domanda vecchia"])
    sent = "\n".join(block.content for block in client.received[0]["input"])
    assert "Domanda vecchia" in sent


def test_quiz_endpoint_remembers_questions_and_passes_them_next_time():
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    from tests.fakes import FakeRetriever, FakeVerifier

    seen = []

    class Gen:
        def generate(self, title, passage, source, n_questions=3, focus=None, avoid=None):
            seen.append(list(avoid or []))
            return [q(f"Domanda {len(seen)}")]

    db = Database(":memory:")
    app = create_app(db, lambda: Gen(), lambda: FakeRetriever(), lambda: FakeVerifier())
    client = TestClient(app)
    client.post("/quiz", json={"title": "Inception", "n_questions": 1})
    client.post("/quiz", json={"title": "Inception", "n_questions": 1})
    assert seen[0] == []
    assert seen[1] == ["Domanda 1"]
