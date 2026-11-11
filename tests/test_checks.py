from src.quiz.checks import is_grounded, matches_expected


def test_is_grounded_true_when_option_words_in_passage():
    passage = "Inception è stato scritto e diretto da Christopher Nolan."
    assert is_grounded("Christopher Nolan", passage)


def test_is_grounded_false_when_option_words_absent():
    passage = "Inception è stato scritto e diretto da Christopher Nolan."
    assert not is_grounded("Steven Spielberg", passage)


def test_is_grounded_false_for_option_without_content_words():
    assert not is_grounded("a di", "Qualsiasi testo")



def test_matches_expected_true_for_same_answer_different_wording():
    assert matches_expected("Il premio Oscar per gli effetti visivi", "Oscar per i migliori effetti visivi")


def test_matches_expected_false_for_different_answer():
    assert not matches_expected("Steven Spielberg", "Christopher Nolan")


def test_matches_expected_false_when_expected_empty():
    assert not matches_expected("Nolan", "")


def test_is_grounded_true_for_short_option_present_in_passage():
    passage = "Breaking Bad è stata trasmessa sulla rete AMC."
    assert is_grounded("AMC", passage)


def test_is_grounded_false_for_short_option_absent_from_passage():
    passage = "Breaking Bad è stata trasmessa sulla rete AMC."
    assert not is_grounded("HBO", passage)
