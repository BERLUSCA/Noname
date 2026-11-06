import pytest

from src.agent import llm


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_AGENT", "test-agent-model")
    monkeypatch.setenv("MODEL_JUDGE", "test-judge-model")


def test_create_client_uses_model_agent(llm_env):
    client = llm.create_client("MODEL_AGENT")
    assert client.model_name == "test-agent-model"


def test_create_client_uses_model_judge(llm_env):
    client = llm.create_client("MODEL_JUDGE")
    assert client.model_name == "test-judge-model"


def test_create_client_returns_anthropic_client(llm_env):
    client = llm.create_client()
    assert type(client).__name__ == "AnthropicClient"


def test_create_client_raises_when_model_missing(llm_env, monkeypatch):
    monkeypatch.delenv("MODEL_AGENT")
    with pytest.raises(RuntimeError, match="MODEL_AGENT"):
        llm.create_client("MODEL_AGENT")


def test_create_client_raises_when_api_key_missing(llm_env, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        llm.create_client()
