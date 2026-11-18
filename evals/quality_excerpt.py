"""Answer quality with long vs short corpus excerpts, judged by the answer judge. Paid calls.

Per question: the judge says correct / wrong / refusal against the expected fact.
The old keyword check is kept for comparison only.
Usage: .venv/Scripts/python.exe evals/quality_excerpt.py --embedding-model <name> [--judge <model>]
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import tools as agent_tools  # noqa: E402
from src.agent.llm import create_client  # noqa: E402
from src.api.db import Database  # noqa: E402
from src.api.main import create_app  # noqa: E402
from src.observability.logger import CallLogger  # noqa: E402
from src.quiz.answer_judge import AnswerJudge  # noqa: E402
from src.quiz.generator import QuizGenerator  # noqa: E402
from src.rag.service import load_retriever_from_env  # noqa: E402
from src.rag.verifier import FactVerifier  # noqa: E402

# (question, keywords for the old check, expected fact for the judge)
QUESTIONS = [
    ("Chi ha diretto Inception?", ["Nolan"], "Inception è stato diretto da Christopher Nolan."),
    ("Chi ha diretto Il padrino?", ["Coppola"], "Il padrino è stato diretto da Francis Ford Coppola."),
    ("Su quale rete andava in onda Breaking Bad?", ["AMC"], "Breaking Bad andava in onda sulla rete AMC."),
    ("Quante stagioni ha Friends?", ["dieci", "10"], "Friends ha dieci stagioni."),
    ("Chi ha diretto Pulp Fiction?", ["Tarantino"], "Pulp Fiction è stato diretto da Quentin Tarantino."),
    ("Chi ha scritto il romanzo da cui è tratto Il miglio verde?", ["King"], "Il miglio verde è tratto dal romanzo di Stephen King."),
]

SETTINGS = {"corto (500 car., 2 chunk)": (500, 2), "lungo (1200 car., 4 chunk)": (1200, 4)}


def ask(chars: int, chunks: int, question: str) -> str:
    agent_tools.EXCERPT_CHARS = chars
    agent_tools.EXCERPT_CHUNKS = chunks
    tmp = Path(tempfile.mkdtemp())
    logger = CallLogger(tmp / "calls.jsonl")
    app = create_app(
        Database(tmp / "q.db"),
        lambda: QuizGenerator(create_client("MODEL_AGENT"), logger=logger),
        load_retriever_from_env,
        lambda: FactVerifier(create_client("MODEL_JUDGE"), logger=logger),
        chat_client_factory=lambda: create_client("MODEL_AGENT"),
        call_logger=logger,
    )
    return TestClient(app).post("/chat", json={"message": question}).json()["answer"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--judge", help="judge model name (MODEL_JUDGE for this run)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    if args.judge:
        os.environ["MODEL_JUDGE"] = args.judge
    judge = AnswerJudge(create_client("MODEL_JUDGE"))

    defaults = (agent_tools.EXCERPT_CHARS, agent_tools.EXCERPT_CHUNKS)
    try:
        for label, (chars, chunks) in SETTINGS.items():
            verdicts = Counter()
            keyword_hits = 0
            for question, keywords, fact in QUESTIONS:
                answer = ask(chars, chunks, question)
                verdict = judge.judge(question, fact, answer).verdict
                keyword_ok = any(k.lower() in answer.lower() for k in keywords)
                verdicts[verdict] += 1
                keyword_hits += keyword_ok
                print(json.dumps({"setting": label, "question": question, "judge": verdict,
                                  "keyword_ok": keyword_ok}, ensure_ascii=False))
            print("SUMMARY", json.dumps({"setting": label, "judge": dict(verdicts),
                                         "keyword_correct": keyword_hits, "total": len(QUESTIONS),
                                         "judge_model": judge.client.model_name}, ensure_ascii=False))
    finally:
        agent_tools.EXCERPT_CHARS, agent_tools.EXCERPT_CHUNKS = defaults


if __name__ == "__main__":
    main()
