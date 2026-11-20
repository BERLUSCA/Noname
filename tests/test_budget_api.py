"""API tests for the budget guardrail: refused calls must not reach the model. No network."""

from fastapi.testclient import TestClient

from src.api.db import Database
from src.api.main import create_app
from src.observability.budget import SessionBudget
from tests.fakes import FakeRetriever, FakeVerifier
from tests.test_agent_chat import ScriptedClient, StubGenerator


class CountingGenerator(StubGenerator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return super().generate(*args, **kwargs)


def make(budget, generator):
    app = create_app(
        Database(":memory:"),
        lambda: generator,
        lambda: FakeRetriever(),
        lambda: FakeVerifier(),
        chat_client_factory=lambda: ScriptedClient([("text", "ok")]),
        budget=budget,
    )
    return TestClient(app)


def exhausted() -> SessionBudget:
    budget = SessionBudget(limit_usd=1.0)
    budget.add(1.0)
    return budget


def test_quiz_refused_when_budget_exhausted_and_model_not_called():
    generator = CountingGenerator()
    response = make(exhausted(), generator).post("/quiz", json={"title": "Inception", "n_questions": 1})
    assert response.status_code == 503
    assert "esaurito" in response.json()["detail"]
    assert generator.calls == 0


def test_chat_refused_when_budget_not_configured():
    response = make(SessionBudget(limit_usd=None), CountingGenerator()).post("/chat", json={"player": "Anna", "message": "ciao"})
    assert response.status_code == 503
    assert "non configurato" in response.json()["detail"]


def test_quiz_allowed_when_budget_has_room():
    generator = CountingGenerator()
    budget = SessionBudget(limit_usd=10.0)
    response = make(budget, generator).post("/quiz", json={"title": "Inception", "n_questions": 1})
    assert response.status_code == 200
    assert generator.calls == 1
