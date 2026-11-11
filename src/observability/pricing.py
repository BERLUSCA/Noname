"""Price table per model, in USD per million tokens. Single source for cost estimates.

Source: https://platform.claude.com/docs/en/about-claude/pricing
Verified on: 2026-11-06 (standard prices, prompts up to 100,000 tokens, no batch, no caching).
Re-verify before the final evaluation.
"""

from dataclasses import dataclass

PRICING_VERIFIED_ON = "2026-11-06"
PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50),
    "claude-sonnet-5-5": ModelPrice(input_per_mtok=2.00, output_per_mtok=10.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated cost in USD. Raises KeyError if the model is not in the price table."""
    price = PRICES[model]
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
