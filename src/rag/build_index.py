"""Builds the vector index from data/raw/ into data/index/, with a fingerprint of corpus, model and chunking.

Usage: .venv/Scripts/python.exe -m src.rag.build_index --model <embedding-model-name>
The model can also come from EMBEDDING_MODEL in .env.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag.embedder import SentenceEmbedder
from src.rag.fingerprint import build_fingerprint, save_fingerprint
from src.rag.index import VectorIndex, build_chunks

ROOT = Path(__file__).resolve().parents[2]
MAX_CHARS = 600
OVERLAP = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG index")
    parser.add_argument("--model", help="sentence-transformers model name")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    model_name = args.model or os.environ.get("EMBEDDING_MODEL", "")
    if not model_name:
        raise SystemExit("Specificare --model oppure EMBEDDING_MODEL in .env")

    raw_dir = ROOT / "data" / "raw"
    index_dir = ROOT / "data" / "index"
    chunks = build_chunks(raw_dir, MAX_CHARS, OVERLAP)
    index = VectorIndex.build(chunks, SentenceEmbedder(model_name))
    index.save(index_dir)
    save_fingerprint(build_fingerprint(raw_dir, model_name, MAX_CHARS, OVERLAP), index_dir)
    print(f"Indice creato: {len(chunks)} chunk, dimensione {index.vectors.shape[1]}, modello {model_name}")


if __name__ == "__main__":
    main()
