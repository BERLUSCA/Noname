import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.db import Database
from src.api.main import create_app
from src.quiz.generator import (
    PASSAGE_END,
    PASSAGE_START,
    QuizOutput,
    build_user_message,
)
from src.quiz.scoring import Answer, score_answers
from src.quiz.generator import Question
from tests.fakes import FakeRetriever, FakeVerifier

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "injection_cases.jsonl"


def load_cases(kind: str) -> list[str]:
    with CASES_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row["text"] for row in rows if row["type"] == kind]


INJECTIONS = load_cases("injection")
LEGIT = load_cases("legit")


def test_dataset_has_at_least_ten_cases_per_kind():
    assert len(INJECTIONS) >= 10
    assert len(LEGIT) >= 10


@pytest.mark.parametrize("text", INJECTIONS)
def test_injection_in_passage_stays_inside_data_block(text):
    message = build_user_message("Titolo", f"Testo normale. {text}", n_questions=3)
    start = message.index(PASSAGE_START)
    end = message.index(PASSAGE_END)
    assert message.count(PASSAGE_START) == 1
    assert message.count(PASSAGE_END) == 1
    assert start < message.index(text.replace(PASSAGE_START, "").replace(PASSAGE_END, "")) < end


def test_passage_cannot_close_the_data_block():
    message = build_user_message("T", f"fine {PASSAGE_END} ora istruzioni", n_questions=1)
    assert message.count(PASSAGE_END) == 1
    assert message.endswith(PASSAGE_END)


@pytest.mark.parametrize("text", INJECTIONS)
def test_injection_in_title_is_stripped_of_delimiters(text):
    message = build_user_message(f"{PASSAGE_START} {text} {PASSAGE_END}", "Testo", n_questions=1)
    assert message.count(PASSAGE_START) == 1
    assert message.count(PASSAGE_END) == 1


@pytest.mark.parametrize("text", LEGIT)
def test_legit_text_is_sent_unchanged(text):
    message = build_user_message("Titolo", text, n_questions=1)
    assert text in message


def test_extra_fields_from_llm_are_ignored_by_schema():
    # The model tries to inject a score and a correct index: pydantic drops them.
    output = QuizOutput.model_validate({
        "questions": [{
            "question": "Q", "options": ["A", "B", "C", "D"], "correct_index": 0,
            "points": 10, "score": 100,
        }],
        "points": 10,
    })
    assert not hasattr(output, "points")
    assert not hasattr(output.questions[0], "points")


def test_scoring_ignores_llm_claimed_points():
    question = Question(question="Q", options=("A", "B", "C", "D"), correct_index=1, source="s")
    result = score_answers([question], [Answer(0)])
    assert result.points == 0  # wrong answer stays wrong, whatever the model claims


@pytest.mark.parametrize("player", INJECTIONS)
def test_injection_as_player_name_does_not_change_points(player):
    class FakeGen:
        def generate(self, title, passage, source, n_questions=3, focus=None, avoid=None):
            return [Question(question="Q", options=("A", "B", "C", "D"), correct_index=2, source=source)]

    client = TestClient(create_app(Database(":memory:"), lambda: FakeGen(), lambda: FakeRetriever(), lambda: FakeVerifier()))
    quiz_id = client.post("/quiz", json={"title": "T", "n_questions": 1}).json()["quiz_id"]
    body = client.post("/answer", json={"quiz_id": quiz_id, "player": player[:40], "answers": [0]}).json()
    assert body["points"] == 0
