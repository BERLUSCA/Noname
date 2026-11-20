"""Tests for POST /chat: scripted agent client, fake retriever and judge, no network."""

import json

from fastapi.testclient import TestClient

from src.api.db import Database
from src.agent.chat import UNVERIFIED_NOTE
from src.api.main import create_app
from src.observability.logger import CallLogger
from tests.fakes import FakeRetriever, FakeVerifier
from tests.test_agent_chat import ScriptedClient, StubGenerator


def make_client(script, logger=None, chat_enabled=True):
    app = create_app(
        Database(":memory:"),
        lambda: StubGenerator(),
        lambda: FakeRetriever(),
        lambda: FakeVerifier(),
        chat_client_factory=(lambda: ScriptedClient(script)) if chat_enabled else None,
        call_logger=logger,
    )
    return TestClient(app)


def test_chat_without_document_returns_out_of_scope_message():
    client = make_client([("text", "Ciao, come posso aiutarti?")])
    body = client.post("/chat", json={"player": "Anna", "message": "Cos'è la carbonara?"}).json()
    assert body["answer"].endswith(UNVERIFIED_NOTE)
    assert body["tools_called"] == []


def test_chat_can_create_quiz_and_returns_quiz_id():
    client = make_client([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}), ("text", "Ecco il quiz.")])
    body = client.post("/chat", json={"player": "Anna", "message": "quiz su Inception"}).json()
    assert body["tools_called"] == ["create_quiz"]
    assert len(body["quiz_ids"]) == 1


def test_chat_response_never_contains_correct_index():
    client = make_client([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}), ("text", "Ecco.")])
    assert "correct_index" not in str(client.post("/chat", json={"player": "Anna", "message": "quiz"}).json())


def test_chat_disabled_without_client_factory_returns_503():
    client = make_client([], chat_enabled=False)
    assert client.post("/chat", json={"player": "Anna", "message": "ciao"}).status_code == 503


def test_chat_rejects_empty_message():
    client = make_client([("text", "x")])
    assert client.post("/chat", json={"player": "Anna", "message": ""}).status_code == 422


def test_chat_writes_one_log_record_per_request(tmp_path):
    path = tmp_path / "calls.jsonl"
    client = make_client([("tool", "search_corpus", {"title": "Inception"}), ("text", "Trovato.")],
                         logger=CallLogger(path))
    client.post("/chat", json={"player": "Anna", "message": "cerca Inception"})
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["operation"] == "chat"
    assert record["tools_called"] == ["search_corpus"]


def test_chat_returns_questions_so_the_page_can_show_a_form():
    client = make_client([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}), ("text", "Ecco.")])
    body = client.post("/chat", json={"player": "Anna", "message": "quiz"}).json()
    quiz = body["quizzes"][0]
    assert quiz["quiz_id"] == body["quiz_ids"][0]
    assert len(quiz["questions"]) == 2
    assert all(len(q["options"]) == 4 and q["question"] for q in quiz["questions"])
    assert "correct" not in str(body["quizzes"])


def test_chat_quiz_can_be_answered_with_the_answer_endpoint():
    client = make_client([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}), ("text", "Ecco.")])
    body = client.post("/chat", json={"player": "Anna", "message": "quiz"}).json()
    result = client.post("/answer", json={"quiz_id": body["quizzes"][0]["quiz_id"], "player": "Anna",
                                          "answers": [0, None]}).json()
    assert result["total"] == 2 and len(result["results"]) == 2
    assert result["unanswered"] == 1


def test_chat_without_quiz_returns_no_quizzes():
    client = make_client([("tool", "search_corpus", {"title": "Inception"}), ("text", "Trovato.")])
    assert client.post("/chat", json={"player": "Anna", "message": "cerca"}).json()["quizzes"] == []


def test_chat_returns_the_players_challenges_without_scores():
    db = Database(":memory:")
    db.add_player("Anna")
    db.add_player("Luca")
    db.create_challenge("ABC123", "q", "Anna", invited="Luca")
    app = create_app(db, lambda: StubGenerator(), lambda: FakeRetriever(), lambda: FakeVerifier(),
                     chat_client_factory=lambda: ScriptedClient([("tool", "my_challenges", {}), ("text", "Ecco le tue sfide.")]))
    body = TestClient(app).post("/chat", json={"player": "Luca", "message": "le mie sfide"}).json()
    assert [c["code"] for c in body["challenges"]] == ["ABC123"]
    assert body["challenges"][0]["role"] == "avversario" and body["challenges"][0]["challenger"] == "Anna"
    assert "points" not in str(body) and "correct" not in str(body)


def test_chat_without_listing_challenges_returns_none():
    client = make_client([("tool", "search_corpus", {"title": "Inception"}), ("text", "Trovato.")])
    assert client.post("/chat", json={"player": "Anna", "message": "cerca"}).json()["challenges"] is None


def test_chat_returns_the_accepted_challenge_code():
    db = Database(":memory:")
    db.add_player("Anna")
    db.add_player("Luca")
    db.create_challenge("ABC123", "q", "Anna", invited="Luca")
    app = create_app(db, lambda: StubGenerator(), lambda: FakeRetriever(), lambda: FakeVerifier(),
                     chat_client_factory=lambda: ScriptedClient([("tool", "accept_challenge", {}), ("text", "Ecco.")]))
    body = TestClient(app).post("/chat", json={"player": "Luca", "message": "accetto la sfida"}).json()
    assert body["accept"] == "ABC123"


def test_chat_without_accepting_returns_no_accept_code():
    client = make_client([("tool", "search_corpus", {"title": "Inception"}), ("text", "Trovato.")])
    assert client.post("/chat", json={"player": "Anna", "message": "cerca"}).json()["accept"] is None


def test_chat_agent_receives_the_corpus_titles():
    client = make_client([("tool", "list_titles", {}), ("text", "Ecco i titoli.")])
    body = client.post("/chat", json={"player": "Anna", "message": "che film ci sono?"}).json()
    assert body["tools_called"] == ["list_titles"]
    assert not body["answer"].endswith(UNVERIFIED_NOTE)  # the titles came from the manifest: app data, verified
