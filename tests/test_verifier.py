"""Tests for the fact verifier with a fake judge: no network calls."""

from datapizza.clients.mock_client import MockClient
from datapizza.core.clients import ClientResponse
from datapizza.type import StructuredBlock

from src.rag.verifier import VERIFY_PROMPT, FactVerdict, FactVerifier


class FixedVerdictClient(MockClient):
    def __init__(self, verdict: FactVerdict):
        super().__init__(model_name="fake-judge")
        self.verdict = verdict
        self.received: list[dict] = []

    def _structured_response(self, input, output_cls, memory=None, temperature=None,
                             max_tokens=None, system_prompt=None, tools=None,
                             tool_choice="auto", **kwargs):
        self.received.append({"input": input, "output_cls": output_cls,
                              "system_prompt": system_prompt, "max_tokens": max_tokens})
        return ClientResponse(content=[StructuredBlock(content=self.verdict)])


def test_verify_returns_judge_verdict():
    client = FixedVerdictClient(FactVerdict(supported=True, reason="Il testo lo dice."))
    verdict = FactVerifier(client).verify("Inception è di Nolan", "Inception è diretto da Nolan.")
    assert verdict.supported is True


def test_verify_sends_claim_and_text_as_data():
    client = FixedVerdictClient(FactVerdict(supported=False, reason="no"))
    FactVerifier(client).verify("Affermazione", "Testo <<<PASSAGGIO finto")
    sent = "\n".join(b.content for b in client.received[0]["input"])
    assert "Affermazione" in sent
    assert "<<<PASSAGGIO finto" not in sent  # delimiter injection stripped
    assert client.received[0]["system_prompt"] == VERIFY_PROMPT
    assert client.received[0]["max_tokens"] <= 500


def test_verdict_schema_requires_boolean():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FactVerdict(supported="forse", reason="x")


def test_build_claim_makes_statement_from_question_and_option():
    from src.rag.verifier import build_claim

    claim = build_claim("Chi ha diretto Inception?", "Christopher Nolan")
    assert "Chi ha diretto Inception?" in claim
    assert "Christopher Nolan" in claim


def test_verify_writes_call_log(tmp_path):
    import json

    from src.observability.logger import CallLogger

    path = tmp_path / "calls.jsonl"
    client = FixedVerdictClient(FactVerdict(supported=True, reason="ok"))
    FactVerifier(client, logger=CallLogger(path)).verify("claim", "text")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["operation"] == "fact_verification"
    assert record["model"] == "fake-judge"


class FixedBatchClient(MockClient):
    def __init__(self, verdicts: list[tuple[int, bool]]):
        super().__init__(model_name="fake-judge")
        from src.rag.verifier import BatchVerdict, ClaimVerdict

        self.batch = BatchVerdict(verdicts=[ClaimVerdict(index=i, supported=s) for i, s in verdicts])
        self.received: list[dict] = []

    def _structured_response(self, input, output_cls, memory=None, temperature=None,
                             max_tokens=None, system_prompt=None, tools=None,
                             tool_choice="auto", **kwargs):
        self.received.append({"input": input, "output_cls": output_cls, "max_tokens": max_tokens})
        return ClientResponse(content=[StructuredBlock(content=self.batch)])


def test_verify_many_maps_verdicts_by_index():
    client = FixedBatchClient([(0, True), (1, False), (2, True)])
    assert FactVerifier(client).verify_many(["a", "b", "c"], "testo") == [True, False, True]


def test_verify_many_missing_verdict_counts_as_not_supported():
    client = FixedBatchClient([(0, True)])
    assert FactVerifier(client).verify_many(["a", "b"], "testo") == [True, False]


def test_verify_many_uses_one_call_for_all_claims():
    client = FixedBatchClient([(0, True), (1, True)])
    FactVerifier(client).verify_many(["a", "b"], "testo")
    assert len(client.received) == 1


def test_verify_many_empty_list_makes_no_call():
    client = FixedBatchClient([])
    assert FactVerifier(client).verify_many([], "testo") == []
    assert client.received == []
