import pytest

from src.quiz.generator import Question
from src.quiz.scoring import POINTS_PER_CORRECT, Answer, score_answers


def make_question(correct_index: int) -> Question:
    return Question(
        question="Q",
        options=("A", "B", "C", "D"),
        correct_index=correct_index,
        source="wiki:X",
    )


def test_score_answers_counts_all_correct():
    questions = [make_question(0), make_question(2)]
    result = score_answers(questions, [Answer(0), Answer(2)])
    assert result.correct == 2
    assert result.total == 2
    assert result.points == 2 * POINTS_PER_CORRECT


def test_score_answers_counts_wrong_answers_as_zero():
    questions = [make_question(0), make_question(2)]
    result = score_answers(questions, [Answer(1), Answer(2)])
    assert result.correct == 1
    assert result.points == POINTS_PER_CORRECT
    assert result.results == (False, True)


def test_score_answers_all_wrong_gives_zero_points():
    questions = [make_question(0), make_question(1)]
    result = score_answers(questions, [Answer(3), Answer(3)])
    assert result.points == 0


def test_score_answers_unanswered_gives_zero_points_and_is_counted():
    questions = [make_question(0), make_question(1)]
    result = score_answers(questions, [Answer(0), Answer(None)])
    assert result.points == POINTS_PER_CORRECT
    assert result.unanswered == 1
    assert result.results == (True, False)


def test_score_answers_all_unanswered_gives_zero_points():
    questions = [make_question(0), make_question(1)]
    result = score_answers(questions, [Answer(None), Answer(None)])
    assert result.points == 0
    assert result.unanswered == 2


def test_score_answers_ignores_elapsed_seconds_for_now():
    questions = [make_question(0)]
    fast = score_answers(questions, [Answer(0, elapsed_seconds=1.0)])
    slow = score_answers(questions, [Answer(0, elapsed_seconds=999.0)])
    assert fast.points == slow.points == POINTS_PER_CORRECT


def test_score_answers_raises_when_answer_count_differs():
    with pytest.raises(ValueError, match="Attese 2"):
        score_answers([make_question(0), make_question(1)], [Answer(0)])


@pytest.mark.parametrize("bad_choice", [-1, 4, "0"])
def test_score_answers_raises_when_choice_invalid(bad_choice):
    with pytest.raises(ValueError, match="Risposta non valida"):
        score_answers([make_question(0)], [Answer(bad_choice)])


def test_score_answers_raises_when_elapsed_negative():
    with pytest.raises(ValueError, match="Tempo di risposta non valido"):
        score_answers([make_question(0)], [Answer(0, elapsed_seconds=-1.0)])
