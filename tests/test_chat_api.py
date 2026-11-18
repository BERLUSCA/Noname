"""Tests for POST /chat: scripted agent client, fake retriever and judge, no network."""

import json

from fastapi.testclient import TestClient

from src.api.db import Database
from src.agent.chat import OUT_OF_SCOPE_ANSWER
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
    body = client.post("/chat", json={"message": "Cos'è la carbonara?"}).json()
    assert body["answer"] == OUT_OF_SCOPE_ANSWER
    assert body["tools_called"] == []


def test_chat_can_create_quiz_and_returns_quiz_id():
    client = make_client([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}), ("text", "Ecco il quiz.")])
    body = client.post("/chat", json={"message": "quiz su Inception"}).json()
    assert body["tools_called"] == ["create_quiz"]
    assert len(body["quiz_ids"]) == 1


def test_chat_response_never_contains_correct_index():
    client = make_client([("tool", "create_quiz", {"title": "Inception", "n_questions": 2}), ("text", "Ecco.")])
    assert "correct_index" not in str(client.post("/chat", json={"message": "quiz"}).json())


def test_chat_disabled_without_client_factory_returns_503():
    client = make_client([], chat_enabled=False)
    assert client.post("/chat", json={"message": "ciao"}).status_code == 503


def test_chat_rejects_empty_message():
    client = make_client([("text", "x")])
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_writes_one_log_record_per_request(tmp_path):
    path = tmp_path / "calls.jsonl"
    client = make_client([("tool", "search_corpus", {"title": "Inception"}), ("text", "Trovato.")],
                         logger=CallLogger(path))
    client.post("/chat", json={"message": "cerca Inception"})
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["operation"] == "chat"
    assert record["tools_called"] == ["search_corpus"]
