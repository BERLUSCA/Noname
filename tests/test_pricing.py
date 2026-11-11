import pytest

from src.observability.pricing import PRICES, estimate_cost_usd


def test_estimate_cost_usd_haiku_one_million_each_way():
    cost = estimate_cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.10 + 0.50)


def test_estimate_cost_usd_zero_tokens_is_zero():
    assert estimate_cost_usd("claude-sonnet-5-5", 0, 0) == 0


def test_estimate_cost_usd_raises_for_unknown_model():
    with pytest.raises(KeyError):
        estimate_cost_usd("unknown-model", 1, 1)


def test_price_table_has_required_models():
    assert {"claude-haiku-5-5", "claude-sonnet-5-5"} <= set(PRICES)
