"""Tests for chunking: pure code, no network."""

import pytest

from src.ingestion.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Una frase breve.") == ["Una frase breve."]


def test_paragraphs_are_packed_without_exceeding_max_chars():
    text = "\n".join(["a" * 100] * 10)
    chunks = chunk_text(text, max_chars=250, overlap=20)
    assert all(len(c) <= 250 for c in chunks)
    assert len(chunks) == 5  # two 100-char paragraphs fit per chunk


def test_long_paragraph_is_split_with_overlap():
    text = "x" * 1000
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0][-50:] == chunks[1][:50]


def test_empty_lines_are_ignored():
    assert chunk_text("\n\n  \n") == []


def test_overlap_must_be_smaller_than_max_chars():
    with pytest.raises(ValueError):
        chunk_text("testo", max_chars=10, overlap=10)


def test_clean_text_drops_external_link_lines():
    from src.ingestion.chunking import clean_text

    text = "Inception è un film.\nInception, su IMDb, IMDb.com.\n(EN) Inception, su Comic Vine.\nIl film ha vinto un Oscar."
    assert clean_text(text) == "Inception è un film.\nIl film ha vinto un Oscar."


def test_clean_text_drops_headings_and_boilerplate():
    from src.ingestion.chunking import clean_text

    text = "Un fatto utile.\n== Riconoscimenti ==\nVoci correlate\nSogno\nAltro fatto utile."
    cleaned = clean_text(text)
    assert "==" not in cleaned
    assert "Voci correlate" not in cleaned
    assert "Un fatto utile." in cleaned
