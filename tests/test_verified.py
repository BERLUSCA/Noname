"""Tests for verified quiz generation: fake generator and fake judge, no network."""

import pytest

from src.quiz.generator import Question
from src.quiz.verified import MAX_GENERATION_ATTEMPTS, NotEnoughVerifiedQuestions, VerificationUnavailable, generate_verified


def q(label):
    return Question(question=label, options=("A", "B", "C", "D"), correct_index=0, source="s")


class Gen:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def generate(self, title, passage, source, n_questions=3, focus=None, avoid=None):
        self.calls += 1
        return self.batches.pop(0)[:n_questions]


class Judge:
    def __init__(self):
        self.calls = 0

    def verify_many(self, claims, text):
        self.calls += 1
        return ["BAD" not in c for c in claims]


def test_keeps_only_supported_and_regenerates_missing():
    gen = Gen([[q("BAD 1"), q("Good 1")], [q("Good 2")]])
    result = generate_verified(gen, Judge(), "T", "P", "S", 2)
    assert [x.question for x in result] == ["Good 1", "Good 2"]
    assert gen.calls == 2


def test_raises_after_max_attempts():
    gen = Gen([[q(f"BAD {i}")] for i in range(MAX_GENERATION_ATTEMPTS)])
    with pytest.raises(NotEnoughVerifiedQuestions):
        generate_verified(gen, Judge(), "T", "P", "S", 1)
    assert gen.calls == MAX_GENERATION_ATTEMPTS


def test_judge_failure_raises_verification_unavailable():
    class Broken:
        def verify_many(self, claims, text):
            raise RuntimeError("down")

    with pytest.raises(VerificationUnavailable):
        generate_verified(Gen([[q("Good")]]), Broken(), "T", "P", "S", 1)
