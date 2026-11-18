"""Tests for the session budget: pure code, no network."""

import json

import pytest

from src.observability.budget import BudgetExceeded, SessionBudget
from src.observability.logger import CallLogger
from src.observability.pricing import FALLBACK_PRICE, cost_for_budget


def test_budget_allows_calls_below_limit():
    budget = SessionBudget(limit_usd=1.0)
    budget.add(0.5)
    budget.ensure_available()  # no exception


def test_budget_blocks_once_spent_reaches_limit():
    budget = SessionBudget(limit_usd=1.0)
    budget.add(1.0)
    with pytest.raises(BudgetExceeded, match="esaurito"):
        budget.ensure_available()


def test_budget_without_configuration_blocks_calls():
    with pytest.raises(BudgetExceeded, match="non configurato"):
        SessionBudget(limit_usd=None).ensure_available()


def test_from_env_reads_eur_and_rate(monkeypatch):
    monkeypatch.setenv("BUDGET_EUR", "2")
    monkeypatch.setenv("USD_PER_EUR", "1.10")
    assert SessionBudget.from_env().limit_usd == pytest.approx(2.2)


def test_from_env_without_rate_is_not_configured(monkeypatch):
    monkeypatch.setenv("BUDGET_EUR", "2")
    monkeypatch.delenv("USD_PER_EUR", raising=False)
    assert not SessionBudget.from_env().configured


def test_unknown_model_uses_fallback_price():
    cost, fallback = cost_for_budget("gpt-4o-mini", 1_000_000, 0)
    assert fallback is True
    assert cost == pytest.approx(FALLBACK_PRICE.input_per_mtok)


def test_known_model_is_not_fallback():
    _, fallback = cost_for_budget("claude-haiku-5-5", 1, 1)
    assert fallback is False


def test_logger_adds_cost_to_budget(tmp_path):
    budget = SessionBudget(limit_usd=10.0)
    logger = CallLogger(tmp_path / "c.jsonl", budget=budget)
    logger.log(model="claude-sonnet-5-5", operation="o", request="r", response="s",
               input_tokens=1_000_000, output_tokens=0, latency_s=0.1)
    assert budget.spent_usd == pytest.approx(2.0)


def test_logger_keeps_null_cost_for_unknown_model_but_charges_budget(tmp_path):
    budget = SessionBudget(limit_usd=10.0)
    path = tmp_path / "c.jsonl"
    CallLogger(path, budget=budget).log(model="gpt-4o-mini", operation="o", request="r", response="s",
                                        input_tokens=1_000_000, output_tokens=0, latency_s=0.1)
    assert json.loads(path.read_text(encoding="utf-8"))["cost_usd"] is None
    assert budget.spent_usd == pytest.approx(FALLBACK_PRICE.input_per_mtok)


@pytest.mark.parametrize("text, expected", [("0.91", 0.91), ("0,91", 0.91), ("0,91 EUR", 0.91), ("10", 10.0)])
def test_parse_number_accepts_common_formats(text, expected):
    from src.observability.budget import parse_number

    assert parse_number(text) == pytest.approx(expected)


def test_parse_number_rejects_garbage():
    from src.observability.budget import parse_number

    with pytest.raises(ValueError):
        parse_number("molti")


def test_from_env_accepts_comma_rate_with_unit(monkeypatch):
    monkeypatch.setenv("BUDGET_EUR", "10")
    monkeypatch.setenv("USD_PER_EUR", "0,91 EUR")
    budget = SessionBudget.from_env()
    assert budget.configured
    assert budget.limit_usd == pytest.approx(9.1)
