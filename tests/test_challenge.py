import pytest
from fastapi.testclient import TestClient

from src.api.db import Database
from src.api.main import create_app
from src.quiz.generator import Question
from tests.fakes import FakeRetriever, FakeVerifier


class FakeGenerator:
    """Question 0: correct index 2. Question 1: correct index 0."""

    def generate(self, title, passage, source, n_questions=3, focus=None, avoid=None):
        return [
            Question(question="Q0", options=("A", "B", "C", "D"), correct_index=2, source=source),
            Question(question="Q1", options=("E", "F", "G", "H"), correct_index=0, source=source),
        ][:n_questions]


@pytest.fixture
def client():
    return TestClient(create_app(Database(":memory:"), lambda: FakeGenerator(), lambda: FakeRetriever(), lambda: FakeVerifier()))


@pytest.fixture
def quiz_id(client):
    response = client.post("/quiz", json={"title": "T", "n_questions": 2})
    return response.json()["quiz_id"]


def challenge_by(client, quiz_id, player, answers):
    response = client.post("/challenge", json={"quiz_id": quiz_id, "player": player, "answers": answers})
    assert response.status_code == 201
    return response.json()["code"]


def test_challenge_creation_returns_code(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    assert len(code) == 6


def test_pending_challenge_reveals_no_scores(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    body = client.get(f"/challenge/{code}").json()
    assert body["status"] == "pending"
    assert body["challenger"] == "Anna"
    assert "scores" not in body
    assert "winner" not in body


def test_pending_challenge_contains_no_answers_or_correct_indices(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    text = str(client.get(f"/challenge/{code}").json())
    assert "correct_index" not in text
    assert "source" not in text


def test_opponent_answer_completes_challenge_with_winner(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])  # 2 points
    response = client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 3]})  # 1 point
    body = response.json()
    assert body["status"] == "completed"
    assert body["scores"] == {"Anna": 2, "Luca": 1}
    assert body["winner"] == "Anna"


def test_opponent_response_does_not_contain_challenger_answers(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [0, 0])
    body = client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 0]}).json()
    assert "correct_index" not in str(body)
    assert "answers" not in str(body)
    assert "results" in body["you"]  # only the opponent's own results


def test_completed_challenge_reveals_only_totals(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [0, 0]})
    body = client.get(f"/challenge/{code}").json()
    assert body["status"] == "completed"
    assert body["scores"] == {"Anna": 2, "Luca": 1}
    assert "results" not in str(body)
    assert "correct_index" not in str(body)


def test_equal_scores_give_no_winner(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    body = client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 0]}).json()
    assert body["winner"] is None


def test_challenger_cannot_answer_own_challenge(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    response = client.post(f"/challenge/{code}/answer", json={"player": "Anna", "answers": [2, 0]})
    assert response.status_code == 400


def test_second_opponent_is_rejected_after_acceptance(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 0]})
    response = client.post(f"/challenge/{code}/answer", json={"player": "Marta", "answers": [2, 0]})
    assert response.status_code == 409


def test_opponent_cannot_answer_twice(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 0]})
    response = client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 0]})
    assert response.status_code == 409


def test_unknown_challenge_returns_404(client):
    assert client.get("/challenge/NOPE00").status_code == 404
    assert client.post("/challenge/NOPE00/answer", json={"player": "Luca", "answers": [0]}).status_code == 404


def test_challenge_does_not_leak_into_leaderboard_twice(client, quiz_id):
    code = challenge_by(client, quiz_id, "Anna", [2, 0])
    client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 0]})
    board = client.get("/leaderboard").json()["leaderboard"]
    assert {row["player"]: row["points"] for row in board} == {"Anna": 2, "Luca": 2}
