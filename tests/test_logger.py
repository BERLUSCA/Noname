import json

from src.observability.logger import CallLogger


def test_log_appends_one_json_line_per_call(tmp_path):
    path = tmp_path / "calls.jsonl"
    logger = CallLogger(path)
    logger.log(model="claude-haiku-5-5", operation="quiz_generation", request="req",
               response="res", input_tokens=100, output_tokens=50, latency_s=1.2345678)
    logger.log(model="claude-haiku-5-5", operation="quiz_generation", request="req2",
               response="res2", input_tokens=10, output_tokens=5, latency_s=0.5)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["request"] == "req2"


def test_log_record_has_required_fields(tmp_path):
    path = tmp_path / "calls.jsonl"
    CallLogger(path).log(model="claude-haiku-5-5", operation="quiz_generation", request="r",
                         response="s", input_tokens=1, output_tokens=1, latency_s=0.1)
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ["timestamp", "model", "operation", "request", "response", "input_tokens",
                "output_tokens", "cost_usd", "latency_s", "retrieved_chunks", "tools_called"]:
        assert key in record


def test_log_computes_cost_for_known_model(tmp_path):
    path = tmp_path / "calls.jsonl"
    record = CallLogger(path).log(model="claude-sonnet-5-5", operation="o", request="r", response="s",
                                  input_tokens=1_000_000, output_tokens=0, latency_s=0.1)
    assert record.cost_usd == 2.0


def test_log_cost_is_null_for_unknown_model(tmp_path):
    path = tmp_path / "calls.jsonl"
    record = CallLogger(path).log(model="gpt-4o-mini", operation="o", request="r", response="s",
                                  input_tokens=1, output_tokens=1, latency_s=0.1)
    assert record.cost_usd is None


def test_log_rounds_latency(tmp_path):
    record = CallLogger(tmp_path / "c.jsonl").log(model="m", operation="o", request="r", response="s",
                                                  input_tokens=0, output_tokens=0, latency_s=1.23456)
    assert record.latency_s == 1.235
