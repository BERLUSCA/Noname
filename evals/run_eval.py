import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.llm import create_client  # noqa: E402
from src.observability.logger import CallLogger  # noqa: E402
from src.observability.pricing import PRICES, estimate_cost_usd  # noqa: E402
from src.quiz.checks import is_grounded, matches_expected  # noqa: E402
from src.rag.verifier import FactVerifier, build_claim  # noqa: E402
from src.quiz.generator import QuizGenerator  # noqa: E402

GOLD_PATH = ROOT / "evals" / "gold.jsonl"


def load_gold() -> list[dict]:
    with GOLD_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(generator: QuizGenerator, verifier: FactVerifier, items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        started = time.perf_counter()
        expected = item["options"][item["correct_index"]]
        row = {"id": item["id"], "schema_ok": False, "grounded": False, "matches_gold": False, "verified": False,
               "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "error": None}
        try:
            question = generator.generate(
                item["title"], item["passage"], source=f"gold:{item['id']}", n_questions=1, focus=item["fact"]
            )[0]
            correct_option = question.options[question.correct_index]
            row["schema_ok"] = len(question.options) == 4
            row["grounded"] = is_grounded(correct_option, item["passage"])
            row["matches_gold"] = matches_expected(correct_option, expected)
            # Judge model: does the passage support the marked-correct answer? (one call per item)
            verdict = verifier.verify(build_claim(question.question, correct_option), item["passage"])
            row["verified"] = verdict.supported
        except Exception as exc:  # record and continue: one failed item must not stop the run
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_s"] = round(time.perf_counter() - started, 2)
        usage = generator.last_usage
        if usage is not None and row["error"] is None:
            row["input_tokens"] = usage.prompt_tokens
            row["output_tokens"] = usage.completion_tokens
        rows.append(row)
    return rows


def summarize(rows: list[dict], model: str) -> dict:
    total = len(rows)
    input_tokens = sum(r["input_tokens"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in rows)
    cost = estimate_cost_usd(model, input_tokens, output_tokens) if model in PRICES else None
    return {
        "model": model,
        "items": total,
        "errors": sum(1 for r in rows if r["error"]),
        "schema_ok_rate": round(sum(r["schema_ok"] for r in rows) / total, 2),
        "verified_rate": round(sum(r["verified"] for r in rows) / total, 2),
        "grounded_lexical_rate": round(sum(r["grounded"] for r in rows) / total, 2),
        "matches_gold_rate": round(sum(r["matches_gold"] for r in rows) / total, 2),
        "avg_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6) if cost is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quiz generation eval on the gold set")
    parser.add_argument("--model", help="model name for MODEL_AGENT (overrides .env for this run)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.model:
        os.environ["MODEL_AGENT"] = args.model

    client = create_client("MODEL_AGENT")
    generator = QuizGenerator(client, logger=CallLogger())
    rows = evaluate(generator, FactVerifier(create_client("MODEL_JUDGE"), logger=CallLogger()), load_gold())
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print("SUMMARY", json.dumps(summarize(rows, client.model_name), ensure_ascii=False))


if __name__ == "__main__":
    main()
