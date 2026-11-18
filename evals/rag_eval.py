"""Retrieval eval on the gold set. Local embeddings only (free, no LLM).

Reports:
- lexical_topk: the expected fact is lexically supported by one of the top-k chunks (for the question).
- topical_similarity: cosine similarity between the fact and its best chunk in the document.
  It measures TOPIC, not truth: false facts score high too (see negatives). Not a truth check.
For truth, use evals/run_verifier_eval.py (judge model).

Usage: .venv/Scripts/python.exe evals/rag_eval.py --model <embedding-model> [--top-k 4]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.quiz.checks import is_grounded  # noqa: E402
from src.rag.embedder import SentenceEmbedder  # noqa: E402
from src.rag.index import VectorIndex, best_similar_chunk, retrieve  # noqa: E402


def read_jsonl(name: str) -> list[dict]:
    with (ROOT / "evals" / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    model_name = args.model or os.environ.get("EMBEDDING_MODEL", "")
    if not model_name:
        raise SystemExit("Specificare --model oppure EMBEDDING_MODEL in .env")

    index = VectorIndex.load(ROOT / "data" / "index")
    embedder = SentenceEmbedder(model_name)

    gold = read_jsonl("gold.jsonl")
    lexical_hits = 0
    for item in gold:
        results = retrieve(index, embedder, item["question"], top_k=args.top_k)
        lexical = any(is_grounded(item["fact"], chunk.text) for _, chunk in results)
        similarity, best = best_similar_chunk(index, embedder, item["fact"], item["title"])
        lexical_hits += lexical
        print(json.dumps({"id": item["id"], "lexical_topk": lexical,
                          "topical_similarity": round(similarity, 3), "best_chunk": best.id}, ensure_ascii=False))

    negatives = read_jsonl("rag_negatives.jsonl")
    negative_sims = []
    for item in negatives:
        similarity, best = best_similar_chunk(index, embedder, item["fact"], item["title"])
        negative_sims.append(similarity)
        print(json.dumps({"id": item["id"], "false_fact_similarity": round(similarity, 3),
                          "best_chunk": best.id}, ensure_ascii=False))

    print("SUMMARY", json.dumps({
        "gold_items": len(gold),
        "top_k": args.top_k,
        "lexical_topk_hit_rate": round(lexical_hits / len(gold), 2),
        "false_facts_similarity_min": round(min(negative_sims), 3),
        "false_facts_similarity_max": round(max(negative_sims), 3),
    }))


if __name__ == "__main__":
    main()
