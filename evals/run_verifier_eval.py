"""Live eval of the fact verifier (judge model) on true facts (gold.jsonl) and false facts (rag_negatives.jsonl).

For each claim: the best chunk of its document is checked by the judge. Paid calls (one per claim).
Usage: .venv/Scripts/python.exe evals/run_verifier_eval.py --model <embedding-model> [--judge <judge-model>]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.llm import create_client  # noqa: E402
from src.observability.logger import CallLogger  # noqa: E402
from src.rag.embedder import SentenceEmbedder  # noqa: E402
from src.rag.index import VectorIndex, best_support  # noqa: E402
from src.rag.verifier import FactVerifier  # noqa: E402


def read_jsonl(name: str) -> list[dict]:
    with (ROOT / "evals" / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="embedding model (local, to find the best chunk)")
    parser.add_argument("--judge", help="judge model name for MODEL_JUDGE (overrides .env for this run)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.judge:
        os.environ["MODEL_JUDGE"] = args.judge

    index = VectorIndex.load(ROOT / "data" / "index")
    embedder = SentenceEmbedder(args.model)
    verifier = FactVerifier(create_client("MODEL_JUDGE"))

    cases = [{"id": i["id"], "title": i["title"], "claim": i["fact"], "label": True} for i in read_jsonl("gold.jsonl")]
    cases += [{"id": i["id"], "title": i["title"], "claim": i["fact"], "label": False} for i in read_jsonl("rag_negatives.jsonl")]

    rows = []
    for case in cases:
        _, chunk = best_similar_chunk(index, embedder, case["claim"], case["title"])
        try:
            verdict = verifier.verify(case["claim"], chunk.text)
            predicted = verdict.supported
            error = None
        except Exception as exc:  # keep going
            predicted, error = None, f"{type(exc).__name__}: {exc}"
        rows.append({**case, "predicted": predicted, "correct": predicted == case["label"], "error": error})
        print(json.dumps({k: rows[-1][k] for k in ("id", "label", "predicted", "correct", "error")}, ensure_ascii=False))

    scored = [r for r in rows if r["predicted"] is not None]
    true_cases = [r for r in scored if r["label"]]
    false_cases = [r for r in scored if not r["label"]]
    summary = {
        "judge": verifier.client.model_name,
        "cases": len(rows),
        "errors": len(rows) - len(scored),
        "accuracy": round(sum(r["correct"] for r in scored) / len(scored), 2),
        "false_negative_rate": round(sum(not r["predicted"] for r in true_cases) / len(true_cases), 2),
        "false_positive_rate": round(sum(r["predicted"] for r in false_cases) / len(false_cases), 2),
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
