"""Tests for players, named challenges and chat ownership. No network: fake generator, retriever, judge."""

from fastapi.testclient import TestClient

from src.api.db import Database, player_key
from src.api.main import create_app
from tests.fakes import FakeRetriever, FakeVerifier
from tests.test_agent_chat import ScriptedClient
from tests.test_api import FakeGenerator


def make(chat_script=None):
    db = Database(":memory:")
    app = create_app(db, lambda: FakeGenerator(), lambda: FakeRetriever(), lambda: FakeVerifier(),
                     chat_client_factory=(lambda: ScriptedClient(chat_script or [("text", "ok")])) if chat_script is not None else None)
    return TestClient(app), db


def register(client, *names):
    for name in names:
        assert client.post("/players", json={"name": name}).status_code == 200


def quiz(client):
    return client.post("/quiz", json={"title": "Inception", "n_questions": 2}).json()["quiz_id"]


def test_register_is_idempotent_and_case_insensitive():
    client, _ = make()
    assert client.post("/players", json={"name": "Anna"}).json() == {"name": "Anna", "created": True}
    assert client.post("/players", json={"name": "anna"}).json()["created"] is False
    assert client.get("/players").json() == {"players": ["Anna"]}


def test_names_match_accents_and_case():
    assert player_key("  Èva   Rossi ") == player_key("èva rossi")


def test_named_challenge_requires_registered_opponent():
    client, _ = make()
    register(client, "Anna")
    qid = quiz(client)
    response = client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca",
                                                "answers": [0, 0]})
    assert response.status_code == 400


def test_named_challenge_requires_registered_challenger():
    client, _ = make()
    register(client, "Luca")
    qid = quiz(client)
    response = client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca",
                                                "answers": [0, 0]})
    assert response.status_code == 400


def test_cannot_challenge_yourself():
    client, _ = make()
    register(client, "Anna")
    qid = quiz(client)
    response = client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "anna",
                                                "answers": [0, 0]})
    assert response.status_code == 400


def test_invited_opponent_sees_code_without_scores_and_answers():
    client, _ = make()
    register(client, "Anna", "Luca")
    qid = quiz(client)
    code = client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca",
                                            "answers": [2, 0]}).json()["code"]
    listing = client.get("/players/Luca/challenges", headers={"X-Player": "Luca"}).json()["challenges"]
    assert [c["code"] for c in listing] == [code]
    assert listing[0]["role"] == "avversario"
    assert "points" not in str(listing) and "scores" not in str(listing)
    assert "correct" not in str(listing)


def test_only_invited_opponent_can_answer():
    client, _ = make()
    register(client, "Anna", "Luca", "Marta")
    qid = quiz(client)
    code = client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca",
                                            "answers": [2, 0]}).json()["code"]
    assert client.post(f"/challenge/{code}/answer", json={"player": "Marta", "answers": [2, 0]}).status_code == 403
    assert client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [2, 0]}).status_code == 200


def test_challenges_for_player_include_sent_and_received():
    client, _ = make()
    register(client, "Anna", "Luca")
    qid = quiz(client)
    client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca", "answers": [0, 0]})
    assert len(client.get("/players/Anna/challenges", headers={"X-Player": "Anna"}).json()["challenges"]) == 1
    assert client.get("/players/Anna/challenges", headers={"X-Player": "Anna"}).json()["challenges"][0]["role"] == "sfidante"


def test_chat_requires_player():
    client, _ = make([("text", "ok")])
    assert client.post("/chat", json={"message": "ciao"}).status_code == 422


def test_chat_session_belongs_to_one_player():
    client, _ = make([("text", "a"), ("text", "b")])
    session = client.post("/chat", json={"player": "Anna", "message": "Chi ha diretto Inception?"}).json()["session_id"]
    response = client.post("/chat", json={"player": "Luca", "message": "Chi ha diretto Inception?",
                                          "session_id": session})
    assert response.status_code == 403


def test_chat_session_owner_can_continue():
    client, _ = make([("text", "a"), ("text", "b")])
    session = client.post("/chat", json={"player": "Anna", "message": "Chi ha diretto Inception?"}).json()["session_id"]
    response = client.post("/chat", json={"player": "anna", "message": "Chi ha diretto Inception?",
                                          "session_id": session})
    assert response.status_code == 200


def test_agent_tools_list_players_and_own_challenges_without_scores():
    from src.agent.tools import ChatContext, build_tools
    from tests.test_agent_chat import make_ctx

    db = Database(":memory:")
    db.add_player("Anna")
    db.add_player("Luca")
    db.create_challenge("ABC123", "q", "Anna", invited="Luca")
    ctx = make_ctx()
    ctx = ChatContext(db=db, retriever=ctx.retriever, generator=ctx.generator, verifier=ctx.verifier, player="Luca")
    tools = {t.name: t for t in build_tools(ctx)}
    assert "Anna" in tools["list_players"].func()
    text = tools["my_challenges"].func()
    assert "ABC123" in text
    assert "punti" not in text.lower() and "risposte" not in text.lower()


def test_database_created_before_invited_column_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE quizzes (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL);"
        "CREATE TABLE challenges (code TEXT PRIMARY KEY, quiz_id TEXT NOT NULL, challenger TEXT NOT NULL,"
        " opponent TEXT, created_at TEXT NOT NULL);"
    )
    conn.commit()
    conn.close()
    db = Database(path)
    db.create_challenge("XYZ", "q", "Anna", invited="Luca")  # would fail without the migration
    assert db.get_challenge("XYZ")["invited"] == "Luca"


def test_player_sees_only_own_challenges():
    client, _ = make()
    register(client, "Anna", "Luca", "Marta")
    qid = quiz(client)
    client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca", "answers": [0, 0]})
    client.post("/challenges", json={"quiz_id": qid, "challenger": "Marta", "opponent": "Luca", "answers": [0, 0]})
    # Marta does not see Anna's challenge, and Anna does not see Marta's
    marta = client.get("/players/Marta/challenges", headers={"X-Player": "Marta"}).json()["challenges"]
    assert [c["challenger"] for c in marta] == ["Marta"]


def test_other_players_challenges_are_refused():
    client, _ = make()
    register(client, "Anna", "Luca")
    qid = quiz(client)
    client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca", "answers": [0, 0]})
    # Luca declares his own name but asks for Anna's list: refused
    assert client.get("/players/Anna/challenges", headers={"X-Player": "Luca"}).status_code == 403
    # No declared identity: refused
    assert client.get("/players/Anna/challenges").status_code == 403


def test_challenge_list_includes_creation_date():
    client, _ = make()
    register(client, "Anna", "Luca")
    qid = quiz(client)
    client.post("/challenges", json={"quiz_id": qid, "challenger": "Anna", "opponent": "Luca", "answers": [0, 0]})
    item = client.get("/players/Anna/challenges", headers={"X-Player": "Anna"}).json()["challenges"][0]
    assert item["created_at"]
    assert "T" in item["created_at"]  # ISO 8601 timestamp


# ---- Challenges from the chat ----

def _answered_quiz(client, player="Anna"):
    client.post("/players", json={"name": "Anna"})
    client.post("/players", json={"name": "Luca"})
    quiz = client.post("/quiz", json={"title": "Inception", "n_questions": 2}).json()
    client.post("/answer", json={"quiz_id": quiz["quiz_id"], "player": player, "answers": [1, 1]})
    return quiz["quiz_id"]


def _client():
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    from tests.fakes import FakeRetriever, FakeVerifier
    from tests.test_api import FakeGenerator

    return TestClient(create_app(Database(":memory:"), lambda: FakeGenerator(), lambda: FakeRetriever(),
                                 lambda: FakeVerifier()))


def test_challenge_with_an_answered_quiz_creates_a_code_for_the_opponent():
    client = _client()
    quiz_id = _answered_quiz(client)
    response = client.post(f"/quiz/{quiz_id}/challenge", json={"challenger": "Anna", "opponent": "Luca"})
    assert response.status_code == 201
    code = response.json()["code"]
    mine = client.get("/players/Anna/challenges", headers={"X-Player": "Anna"}).json()["challenges"]
    theirs = client.get("/players/Luca/challenges", headers={"X-Player": "Luca"}).json()["challenges"]
    assert [c["code"] for c in mine] == [code] and mine[0]["role"] == "sfidante"
    assert [c["code"] for c in theirs] == [code] and theirs[0]["role"] == "avversario"
    done = client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [0, 0]}).json()
    assert done["status"] == "completed" and done["winner"] == "Luca"  # correct answers are index 0 in the fakes


def test_challenge_with_a_quiz_requires_a_prior_attempt_and_a_valid_opponent():
    client = _client()
    client.post("/players", json={"name": "Anna"})
    client.post("/players", json={"name": "Luca"})
    quiz = client.post("/quiz", json={"title": "Inception", "n_questions": 2}).json()["quiz_id"]
    body = {"challenger": "Anna", "opponent": "Luca"}
    assert client.post(f"/quiz/{quiz}/challenge", json=body).status_code == 400  # Anna has not answered yet
    client.post("/answer", json={"quiz_id": quiz, "player": "Anna", "answers": [1, 1]})
    assert client.post(f"/quiz/{quiz}/challenge", json={"challenger": "Anna", "opponent": "Anna"}).status_code == 400
    assert client.post(f"/quiz/{quiz}/challenge", json={"challenger": "Anna", "opponent": "Nessuno"}).status_code == 400
    assert client.post("/quiz/non-esiste/challenge", json=body).status_code == 404


def test_cannot_challenge_a_player_who_already_answered_the_quiz():
    client = _client()
    quiz = _answered_quiz(client)
    client.post("/answer", json={"quiz_id": quiz, "player": "Luca", "answers": [0, 0]})
    response = client.post(f"/quiz/{quiz}/challenge", json={"challenger": "Anna", "opponent": "Luca"})
    assert response.status_code == 409


def test_challenge_response_never_reveals_correct_answers():
    client = _client()
    quiz = _answered_quiz(client)
    body = client.post(f"/quiz/{quiz}/challenge", json={"challenger": "Anna", "opponent": "Luca"}).json()
    assert "correct" not in str(body)
