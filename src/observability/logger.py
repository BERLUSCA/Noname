"""JSONL log of every LLM call: one line per call, appended to logs/ (git-ignored)."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.observability.budget import SessionBudget
from src.observability.pricing import PRICES, cost_for_budget, estimate_cost_usd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = ROOT / "logs" / "llm_calls.jsonl"


@dataclass
class LLMCallRecord:
    timestamp: str
    model: str
    operation: str
    request: str
    response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    retrieved_chunks: list[str] = field(default_factory=list)  # chunk ids and sources, filled by RAG later
    tools_called: list[str] = field(default_factory=list)  # filled by the agent later


class CallLogger:
    def __init__(self, path: Path = DEFAULT_LOG_PATH, budget: SessionBudget | None = None):
        self.path = Path(path)
        self.budget = budget

    def log(
        self,
        *,
        model: str,
        operation: str,
        request: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        tools_called: list[str] | None = None,
        retrieved_chunks: list[str] | None = None,
    ) -> LLMCallRecord:
        cost = (
            estimate_cost_usd(model, input_tokens, output_tokens)
            if model in PRICES
            else None
        )
        if self.budget is not None:
            charged, _ = cost_for_budget(model, input_tokens, output_tokens)
            self.budget.add(charged)
        record = LLMCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            operation=operation,
            request=request,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 8) if cost is not None else None,
            latency_s=round(latency_s, 3),
            retrieved_chunks=list(retrieved_chunks or []),
            tools_called=list(tools_called or []),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record
