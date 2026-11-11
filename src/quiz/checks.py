import re

MIN_WORD_LEN = 4  # ignores short words (articles, prepositions)
OVERLAP_THRESHOLD = 0.5


def content_words(text: str) -> set[str]:
    """Long words, or all words if the text has none (e.g. short answers like 'AMC')."""
    words = set(re.findall(r"\w+", text.lower()))
    long_words = {w for w in words if len(w) >= MIN_WORD_LEN}
    return long_words or words


def _overlap(words: set[str], text: str) -> float:
    if not words:
        return 0.0
    text_tokens = set(re.findall(r"\w+", text.lower()))
    return len(words & text_tokens) / len(words)


def is_grounded(option: str, passage: str) -> bool:
    """True if most content words of the option appear in the passage."""
    return _overlap(content_words(option), passage) >= OVERLAP_THRESHOLD



def matches_expected(option: str, expected: str) -> bool:
    """True if the option and the expected answer share most of their content words (both ways)."""
    words = content_words(option)
    expected_words = content_words(expected)
    if not words or not expected_words:
        return False
    shared = len(words & expected_words)
    return shared / len(words) >= OVERLAP_THRESHOLD and shared / len(expected_words) >= OVERLAP_THRESHOLD
