"""Interactive check of the index: prints the top chunks for a query, with sources.

Usage: .venv/Scripts/python.exe -m src.rag.query "domanda" [--model <name>] [--top-k 4]
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag.embedder import SentenceEmbedder
from src.rag.index import VectorIndex, retrieve

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index")
    parser.add_argument("query")
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    model_name = args.model or os.environ.get("EMBEDDING_MODEL", "")
    if not model_name:
        raise SystemExit("Specificare --model oppure EMBEDDING_MODEL in .env")

    index = VectorIndex.load(ROOT / "data" / "index")
    for score, chunk in retrieve(index, SentenceEmbedder(model_name), args.query, args.top_k):
        print(f"{score:.3f}  {chunk.id}  ({chunk.title})")
        print(f"       {chunk.text[:160]}...")


if __name__ == "__main__":
    main()
