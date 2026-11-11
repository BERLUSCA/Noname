from dataclasses import dataclass

from src.quiz.generator import Question

POINTS_PER_CORRECT = 1


@dataclass(frozen=True)
class Answer:
    """A player's answer. choice=None means not answered (0 points).

    elapsed_seconds is reserved for future per-answer timers; scoring ignores it for now.
    """

    choice: int | None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class ScoreResult:
    correct: int
    unanswered: int
    total: int
    points: int
    results: tuple[bool, ...]


def score_answers(questions: list[Question], answers: list[Answer]) -> ScoreResult:
    """Compares each answer with the correct index. Unanswered questions score 0.

    Raises ValueError on invalid input.
    """
    if len(answers) != len(questions):
        raise ValueError(f"Attese {len(questions)} risposte, ricevute {len(answers)}")

    results = []
    unanswered = 0
    for question, answer in zip(questions, answers):
        if answer.elapsed_seconds is not None and answer.elapsed_seconds < 0:
            raise ValueError(f"Tempo di risposta non valido: {answer.elapsed_seconds!r}")
        if answer.choice is None:
            unanswered += 1
            results.append(False)
            continue
        if not isinstance(answer.choice, int) or not 0 <= answer.choice < len(question.options):
            raise ValueError(f"Risposta non valida: {answer.choice!r}")
        results.append(answer.choice == question.correct_index)

    correct = sum(results)
    return ScoreResult(
        correct=correct,
        unanswered=unanswered,
        total=len(questions),
        points=correct * POINTS_PER_CORRECT,
        results=tuple(results),
    )
