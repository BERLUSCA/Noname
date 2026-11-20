"""Tests for judge verification in POST /quiz: no network, fake generator and fake judge."""

from fastapi.testclient import TestClient

from src.api.db import Database
from src.api.main import create_app
from src.quiz.generator import Question
from src.quiz.verified import MAX_GENERATION_ATTEMPTS
from tests.fakes import FakeRetriever


def make_question(label: str) -> Question:
    return Question(question=label, options=("A", "B", "C", "D"), correct_index=0, source="s")


class SequenceGenerator:
    """Returns the given batches of questions, one batch per call; records how many calls were made."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def generate(self, title, passage, source, n_questions=3, focus=None, avoid=None):
        self.calls += 1
        return self.batches.pop(0)[:n_questions]


class ScriptedVerifier:
    def __init__(self):
        self.calls = 0

    def verify_many(self, claims, text):
        self.calls += 1
        return ["BAD" not in c for c in claims]


def build(generator, verifier):
    db = Database(":memory:")
    app = create_app(db, lambda: generator, lambda: FakeRetriever(), lambda: verifier)
    return TestClient(app), db


def test_rejected_question_is_regenerated():
    generator = SequenceGenerator([
        [make_question("BAD 1"), make_question("Good 1")],
        [make_question("Good 2")],
    ])
    client, _ = build(generator, ScriptedVerifier())
    body = client.post("/quiz", json={"title": "Inception", "n_questions": 2}).json()
    assert [q["question"] for q in body["questions"]] == ["Good 1", "Good 2"]
    assert generator.calls == 2


def test_quiz_fails_when_not_enough_verified_questions_after_retry():
    generator = SequenceGenerator([[make_question(f"BAD {i}")] for i in range(MAX_GENERATION_ATTEMPTS)])
    client, db = build(generator, ScriptedVerifier())
    response = client.post("/quiz", json={"title": "Inception", "n_questions": 1})
    assert response.status_code == 503
    assert generator.calls == MAX_GENERATION_ATTEMPTS  # never more than the limit


def test_no_quiz_is_saved_when_verification_fails():
    generator = SequenceGenerator([[make_question(f"BAD {i}")] for i in range(MAX_GENERATION_ATTEMPTS)])
    client, db = build(generator, ScriptedVerifier())
    client.post("/quiz", json={"title": "Inception", "n_questions": 1})
    count = db.conn.execute("SELECT COUNT(*) FROM quizzes").fetchone()[0]
    assert count == 0


def test_judge_called_once_per_attempt_not_per_question():
    verifier = ScriptedVerifier()
    generator = SequenceGenerator([[make_question("Good 1"), make_question("Good 2"), make_question("Good 3")]])
    client, _ = build(generator, verifier)
    client.post("/quiz", json={"title": "Inception", "n_questions": 3})
    assert verifier.calls == 1


def test_judge_error_returns_503():
    class BrokenVerifier:
        def verify_many(self, claims, text):
            raise RuntimeError("giudice down")

    generator = SequenceGenerator([[make_question("Good 1")]])
    client, _ = build(generator, BrokenVerifier())
    assert client.post("/quiz", json={"title": "Inception", "n_questions": 1}).status_code == 503
