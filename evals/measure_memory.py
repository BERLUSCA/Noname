"""Measures input tokens of a 10-turn chat, with the sliding window vs full history. Paid calls.

Usage: .venv/Scripts/python.exe evals/measure_memory.py --embedding-model <name>
Each mode runs the same conversation in a fresh session and database.
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
from src.api import main as api_main  # noqa: E402
from src.api.db import Database  # noqa: E402
from src.observability.logger import CallLogger  # noqa: E402
from src.quiz.generator import QuizGenerator  # noqa: E402
from src.rag.service import load_retriever_from_env  # noqa: E402
from src.rag.verifier import FactVerifier  # noqa: E402

TURNS = [
    "Parlami del film Inception.",
    "Chi lo ha diretto?",
    "In quale anno è uscito?",
    "Ha vinto premi?",
    "E il film Il padrino?",
    "Chi è il regista?",
    "Ha vinto l'Oscar?",
    "Parlami di Breaking Bad.",
    "Su quale rete andava in onda?",
    "Quante stagioni ha?",
]


def run_mode(label: str, window: int, embedding_model: str) -> dict:
    original = api_main.WINDOW_SIZE
    api_main.WINDOW_SIZE = window
    try:
        tmp = Path(tempfile.mkdtemp())
        logger = CallLogger(tmp / "calls.jsonl")
        app = api_main.create_app(
            Database(tmp / "chat.db"),
            lambda: QuizGenerator(create_client("MODEL_AGENT"), logger=logger),
            load_retriever_from_env,
            lambda: FactVerifier(create_client("MODEL_JUDGE"), logger=logger),
            chat_client_factory=lambda: create_client("MODEL_AGENT"),
            call_logger=logger,
        )
        client = TestClient(app)
        session = None
        for text in TURNS:
            body = {"player": "Anna", "message": text}
            if session:
                body["session_id"] = session
            session = client.post("/chat", json=body).json()["session_id"]
        records = [json.loads(line) for line in (tmp / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        chat = [r for r in records if r["operation"] == "chat"]
        per_turn = [r["input_tokens"] for r in chat]
        return {"mode": label, "window": window, "turns": len(chat),
                "input_tokens_total": sum(per_turn), "input_tokens_per_turn": per_turn}
    finally:
        api_main.WINDOW_SIZE = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    windowed = run_mode("finestra 6", 6, args.embedding_model)
    full = run_mode("storia completa", 10_000, args.embedding_model)
    for r in (windowed, full):
        print(json.dumps(r, ensure_ascii=False))
    saving = 1 - windowed["input_tokens_total"] / full["input_tokens_total"]
    print("SUMMARY", json.dumps({"window_total": windowed["input_tokens_total"],
                                 "full_total": full["input_tokens_total"],
                                 "saving_pct": round(saving * 100, 1)}))


if __name__ == "__main__":
    main()
