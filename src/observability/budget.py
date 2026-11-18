"""Session budget (guardrail 7): stop calling paid models when the estimated spend exceeds the limit.

The session is the server process: the counter starts at zero on each restart.
The limit comes from .env: BUDGET_EUR and USD_PER_EUR (exchange rate, to be set by the user and dated).
If either is missing, the budget is not configured and paid calls are refused (fail closed).
"""

import os
import re


def parse_number(text: str) -> float:
    """Reads '0.91', '0,91' or '0,91 EUR' as a number. Anything else raises ValueError."""
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*[A-Za-z]*\s*", text)
    if not match:
        raise ValueError(f"Numero non valido: {text!r}")
    return float(match.group(1).replace(",", "."))


class BudgetExceeded(Exception):
    """Raised when the session has no budget left, or the budget is not configured."""


class SessionBudget:
    def __init__(self, limit_usd: float | None):
        self.limit_usd = limit_usd
        self.spent_usd = 0.0

    @classmethod
    def from_env(cls) -> "SessionBudget":
        try:
            eur = parse_number(os.environ["BUDGET_EUR"])
            usd_per_eur = parse_number(os.environ["USD_PER_EUR"])
        except (KeyError, ValueError):
            return cls(limit_usd=None)
        return cls(limit_usd=eur * usd_per_eur)

    @property
    def configured(self) -> bool:
        return self.limit_usd is not None

    def add(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd

    def ensure_available(self) -> None:
        if not self.configured:
            raise BudgetExceeded("Budget non configurato: imposta BUDGET_EUR e USD_PER_EUR in .env")
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(f"Budget di sessione esaurito ({self.spent_usd:.4f} USD su {self.limit_usd:.4f})")
