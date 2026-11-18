"""Measures repeated questions over N quizzes on one title: before (fixed passage, no memory) vs after
(themed passages + memory of asked questions). Paid calls: N generations per mode.

Repeat = a question whose content words overlap >= 60% with an earlier question on the same title.
Usage: .venv/Scripts/python.exe evals/measure_repetition.py --embedding-model <name> [--title Inception] [--quizzes 10]
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.llm import create_client  # noqa: E402
from src.api.db import Database  # noqa: E402
from src.quiz.generator import QuizGenerator  # noqa: E402
from src.rag.embedder import SentenceEmbedder  # noqa: E402
from src.rag.index import VectorIndex, normalize_title, retrieve_for_title, retrieve_for_themes  # noqa: E402

STOP = {"che", "chi", "come", "quale", "quali", "dove", "quando", "cosa", "nel", "del", "della", "dei", "delle", "il", "la", "lo", "gli", "le", "di", "da", "in", "un", "una", "e", "è", "si"}


TITLE_WORDS: set[str] = set()


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower())
            if len(w) > 2 and w not in STOP and w not in TITLE_WORDS}


def is_repeat(question: str, earlier: list[str]) -> bool:
    w = words(question)
    for old in earlier:
        ow = words(old)
        if w and ow and len(w & ow) / min(len(w), len(ow)) >= 0.6:
            return True
    return False


ASKED: dict[str, list[str]] = {}


def run(mode: str, title: str, quizzes: int, generator: QuizGenerator, index, embedder, db, rng) -> int:
    asked: list[str] = []
    ASKED[mode] = asked
    repeats = 0
    for _ in range(quizzes):
        if mode == "prima":
            chunks = [c for _, c in retrieve_for_title(index, embedder, title, 4)]
            avoid = None
        else:
            from src.rag.service import TitleRetriever
            key = normalize_title(title)
            chunks = TitleRetriever(index, embedder, rng=rng).retrieve(title)
            avoid = db.recent_questions(key)
        passage = "\n".join(c.text for c in chunks)
        for q in generator.generate(title, passage, source=chunks[0].url, n_questions=2, avoid=avoid):
            if is_repeat(q.question, asked):
                repeats += 1
            asked.append(q.question)
            if mode != "prima":
                db.remember_questions(normalize_title(title), [q.question])
    return repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--title", default="Inception")
    parser.add_argument("--quizzes", type=int, default=10)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    TITLE_WORDS.update(re.findall(r"\w+", args.title.lower()))
    index = VectorIndex.load(ROOT / "data" / "index")
    embedder = SentenceEmbedder(args.embedding_model)
    generator = QuizGenerator(create_client("MODEL_AGENT"))
    import tempfile
    db = Database(Path(tempfile.mkdtemp()) / "rep.db")
    before = run("prima", args.title, args.quizzes, generator, index, embedder, db, random.Random(0))
    after = run("dopo", args.title, args.quizzes, generator, index, embedder, Database(Path(tempfile.mkdtemp()) / "rep2.db"), random.Random(0))
    total = args.quizzes * 2
    print("SUMMARY", json.dumps({"title": args.title, "quizzes": args.quizzes, "questions_per_mode": total,
                                 "repeats_before": before, "repeats_after": after}, ensure_ascii=False))
    print("ASKED_BEFORE", json.dumps(ASKED["prima"], ensure_ascii=False))
    print("ASKED_AFTER", json.dumps(ASKED["dopo"], ensure_ascii=False))


if __name__ == "__main__":
    main()
