"""Live end-to-end smoke test of the API with the real model and a temporary database. Paid calls.

Flow: POST /quiz -> POST /challenge (challenger) -> GET pending -> POST /challenge/{code}/answer -> GET completed.
Usage: .venv/Scripts/python.exe evals/smoke_api.py [--model <model-name>]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.llm import create_client  # noqa: E402
from src.api.db import Database  # noqa: E402
from src.api.main import create_app  # noqa: E402
from src.observability.logger import CallLogger  # noqa: E402
from src.quiz.generator import QuizGenerator  # noqa: E402
from src.rag.service import load_retriever_from_env  # noqa: E402
from src.rag.verifier import FactVerifier  # noqa: E402



def check(condition: bool, label: str) -> None:
    print(("OK   " if condition else "FAIL ") + label)
    if not condition:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--embedding-model", help="sentence-transformers model (overrides EMBEDDING_MODEL)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
    if args.model:
        os.environ["MODEL_AGENT"] = args.model

    tmp = tempfile.mkdtemp()
    generator = QuizGenerator(create_client("MODEL_AGENT"), logger=CallLogger())
    verifier = FactVerifier(create_client("MODEL_JUDGE"), logger=CallLogger())
    client = TestClient(create_app(Database(Path(tmp) / "smoke.db"), lambda: generator, load_retriever_from_env, lambda: verifier))

    quiz = client.post("/quiz", json={"title": "Inception", "n_questions": 2}).json()
    check(len(quiz["questions"]) == 2, "quiz creato con 2 domande")
    check("correct_index" not in str(quiz), "domande senza risposta corretta")

    # Challenger answers all with index 0; opponent with index 1.
    code = client.post("/challenge", json={"quiz_id": quiz["quiz_id"], "player": "Anna", "answers": [0, 0]}).json()["code"]
    pending = client.get(f"/challenge/{code}").json()
    check(pending["status"] == "pending" and "scores" not in pending, "sfida pending senza punteggi")
    done = client.post(f"/challenge/{code}/answer", json={"player": "Luca", "answers": [1, 1]}).json()
    check(done["status"] == "completed" and set(done["scores"]) == {"Anna", "Luca"}, "sfida completata con due punteggi")
    check("correct_index" not in str(done), "risposta avversario senza risposte corrette")

    board = client.get("/leaderboard").json()["leaderboard"]
    check(len(board) == 2, "classifica con due giocatori")
    check(all(s.startswith("https://it.wikipedia.org/") for s in done.get("sources", [])) or "sources" not in done, "fonti da Wikipedia")
    print("INFO scores", done["scores"], "winner", done["winner"])
    print("INFO sources", json.dumps(client.post("/answer", json={"quiz_id": quiz["quiz_id"], "player": "Terzo", "answers": [0, 0]}).json().get("sources")))
    print("DONE: API smoke test passed")


if __name__ == "__main__":
    main()
