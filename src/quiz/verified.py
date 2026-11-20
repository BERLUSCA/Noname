"""Verified quiz generation: generate, judge all questions in one call, keep the supported ones.

Shared by POST /quiz and by the agent's create_quiz tool, so both behave the same way.
"""

from src.quiz.generator import Question, QuizGenerator
from src.rag.verifier import FactVerifier, build_claim

MAX_GENERATION_ATTEMPTS = 3  # guardrail 6: at most 3 generation calls + 3 judge calls per quiz


class VerificationUnavailable(Exception):
    """The judge failed; the quiz must not be served unverified."""


class NotEnoughVerifiedQuestions(Exception):
    """Not enough questions survived verification after the allowed attempts."""


def generate_verified(
    generator: QuizGenerator,
    verifier: FactVerifier,
    title: str,
    passage: str,
    source: str,
    n_questions: int,
    avoid: list[str] | None = None,
) -> list[Question]:
    accepted: list[Question] = []
    for _ in range(MAX_GENERATION_ATTEMPTS):
        missing = n_questions - len(accepted)
        if missing <= 0:
            break
        candidates = generator.generate(title, passage, source=source, n_questions=missing, avoid=avoid)
        claims = [build_claim(q.question, q.options[q.correct_index]) for q in candidates]
        try:
            verdicts = verifier.verify_many(claims, passage)
        except Exception as exc:
            raise VerificationUnavailable(str(exc)) from exc
        accepted += [q for q, ok in zip(candidates, verdicts) if ok]
    if len(accepted) < n_questions:
        raise NotEnoughVerifiedQuestions(f"{len(accepted)} domande verificate su {n_questions}")
    return accepted[:n_questions]
