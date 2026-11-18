"""Live smoke test of POST /chat with the real agent model. Paid calls (a few per run).

Usage: .venv/Scripts/python.exe evals/smoke_chat.py --embedding-model <name> [--message "..."]
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--message", default="Fammi un quiz di 2 domande su Inception.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ["EMBEDDING_MODEL"] = args.embedding_model

    logger = CallLogger()
    app = create_app(
        Database(Path(tempfile.mkdtemp()) / "chat.db"),
        lambda: QuizGenerator(create_client("MODEL_AGENT"), logger=logger),
        load_retriever_from_env,
        lambda: FactVerifier(create_client("MODEL_JUDGE"), logger=logger),
        chat_client_factory=lambda: create_client("MODEL_AGENT"),
        call_logger=logger,
    )
    response = TestClient(app).post("/chat", json={"message": args.message})
    print("STATUS", response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
