"""Tests for chunk index and retrieval with a fake embedder: no model download, no network."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.rag.index import Chunk, VectorIndex, build_chunks


class KeywordEmbedder:
    """Toy embedder: one dimension per keyword. Deterministic and offline."""

    KEYWORDS = ["nolan", "amc", "sogno", "rete"]

    def embed(self, texts):
        return np.array([[float(k in t.lower()) for k in self.KEYWORDS] for t in texts])


def make_chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"doc#{i}", slug="doc", title="Doc", url="https://example.org", text=text)


def test_search_returns_most_similar_chunk_first():
    chunks = [make_chunk(0, "Christopher Nolan dirige il film."),
              make_chunk(1, "Va in onda sulla rete AMC."),
              make_chunk(2, "Il protagonista entra nel sogno.")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    results = index.search(KeywordEmbedder().embed(["rete AMC"])[0], top_k=3)
    assert results[0][1].id == "doc#1"


def test_search_rejects_top_k_out_of_range():
    chunks = [make_chunk(i, "nolan") for i in range(5)]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    with pytest.raises(ValueError, match="top_k"):
        index.search(np.array([1.0, 0, 0, 0]), top_k=2)
    with pytest.raises(ValueError, match="top_k"):
        index.search(np.array([1.0, 0, 0, 0]), top_k=6)


def test_index_length_mismatch_raises():
    with pytest.raises(ValueError):
        VectorIndex([make_chunk(0, "x")], np.zeros((2, 4)))


def test_save_and_load_round_trip(tmp_path):
    chunks = [make_chunk(0, "nolan"), make_chunk(1, "amc")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert [c.id for c in loaded.chunks] == ["doc#0", "doc#1"]
    assert np.allclose(loaded.vectors, index.vectors)


def test_build_chunks_reads_manifest_and_text(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps([
        {"slug": "inception", "title": "Inception", "url": "https://it.wikipedia.org/wiki/Inception"}
    ]), encoding="utf-8")
    (tmp_path / "inception.txt").write_text(
        "Paragrafo uno: un fatto abbastanza lungo da superare la soglia minima dei chunk.\nParagrafo due.",
        encoding="utf-8",
    )
    chunks = build_chunks(tmp_path, max_chars=600, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].id == "inception#0"
    assert chunks[0].url == "https://it.wikipedia.org/wiki/Inception"


def test_retrieve_embeds_query_and_returns_sources():
    from src.rag.index import retrieve

    chunks = [make_chunk(0, "Christopher Nolan dirige il film."), make_chunk(1, "Va in onda sulla rete AMC.")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    results = retrieve(index, KeywordEmbedder(), "rete AMC", top_k=3)
    assert results[0][1].url == "https://example.org"
    assert results[0][1].id == "doc#1"


def test_retrieve_for_title_only_returns_chunks_of_that_title():
    from src.rag.index import retrieve_for_title

    chunks = [
        Chunk(id="a#0", slug="a", title="Inception", url="u1", text="nolan"),
        Chunk(id="b#0", slug="b", title="Friends", url="u2", text="nolan amc"),
    ]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    results = retrieve_for_title(index, KeywordEmbedder(), "inception")
    assert [c.id for _, c in results] == ["a#0"]


def test_retrieve_for_title_matches_accents_and_case():
    from src.rag.index import retrieve_for_title

    chunks = [Chunk(id="p#0", slug="p", title="Il padrino (film)", url="u", text="nolan")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    assert retrieve_for_title(index, KeywordEmbedder(), "IL PADRINO")[0][1].id == "p#0"


def test_retrieve_for_title_raises_when_title_missing():
    from src.rag.index import InsufficientMaterial, retrieve_for_title

    chunks = [Chunk(id="a#0", slug="a", title="Inception", url="u", text="nolan")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    with pytest.raises(InsufficientMaterial):
        retrieve_for_title(index, KeywordEmbedder(), "Ricetta della pizza")


def test_retrieve_for_title_accepts_whole_word_prefix():
    from src.rag.index import retrieve_for_title

    chunks = [Chunk(id="l#0", slug="l", title="Il Signore degli Anelli - La Compagnia dell'Anello", url="u", text="nolan")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    assert retrieve_for_title(index, KeywordEmbedder(), "Il Signore Degli Anelli")[0][1].id == "l#0"


def test_retrieve_for_title_rejects_partial_word_prefix():
    from src.rag.index import InsufficientMaterial, retrieve_for_title

    chunks = [Chunk(id="f#0", slug="f", title="Friends", url="u", text="nolan")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    with pytest.raises(InsufficientMaterial):
        retrieve_for_title(index, KeywordEmbedder(), "Frie")


def test_retrieve_for_title_applies_alias():
    from src.rag.index import retrieve_for_title

    chunks = [Chunk(id="p#0", slug="p", title="Psyco", url="u", text="nolan")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    assert retrieve_for_title(index, KeywordEmbedder(), "Psycho")[0][1].id == "p#0"


def test_best_similar_chunk_searches_whole_document_not_only_top_k():
    from src.rag.index import best_similar_chunk

    chunks = [Chunk(id=f"x#{i}", slug="x", title="Film", url="u", text="rete") for i in range(6)]
    chunks.append(Chunk(id="x#6", slug="x", title="Film", url="u", text="nolan"))
    index = VectorIndex.build(chunks, KeywordEmbedder())
    score, chunk = best_similar_chunk(index, KeywordEmbedder(), "nolan", "Film")
    assert chunk.id == "x#6"
    assert score > 0.9



def test_build_chunks_drops_fragments_shorter_than_minimum(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps([
        {"slug": "x", "title": "X", "url": "u"}
    ]), encoding="utf-8")
    (tmp_path / "x.txt").write_text("oduzioni.\n" + "Un fatto informativo abbastanza lungo da tenere nell'indice. " * 2,
                                    encoding="utf-8")
    chunks = build_chunks(tmp_path)
    assert all(len(c.text) >= 80 for c in chunks)
    assert not any(c.text == "oduzioni." for c in chunks)


class QueryRecorder:
    """Embedder that records the queries it receives, to check which themes a request uses."""

    def __init__(self):
        self.queries = []

    def embed(self, texts):
        self.queries.extend(texts)
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts])


def test_retrieve_for_themes_uses_a_random_subset_of_themes():
    import random

    from src.rag.index import THEMES, retrieve_for_themes

    chunks = [Chunk(id=f"f#{i}", slug="f", title="Film", url="u", text=f"testo {i}") for i in range(30)]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    rec1, rec2 = QueryRecorder(), QueryRecorder()
    picked = retrieve_for_themes(index, rec1, "Film", random.Random(1))
    retrieve_for_themes(index, rec2, "Film", random.Random(7))
    ids = [c.id for _, c in picked]
    assert len(ids) == len(set(ids))  # no duplicate chunks
    assert all(q.startswith("Film ") for q in rec1.queries[1:])
    assert set(rec1.queries[1:]) != set(rec2.queries[1:]) or len(THEMES) == 1


def test_retrieve_for_themes_raises_when_title_missing():
    import random

    from src.rag.index import InsufficientMaterial, retrieve_for_themes

    chunks = [Chunk(id="a#0", slug="a", title="Inception", url="u", text="nolan")]
    index = VectorIndex.build(chunks, KeywordEmbedder())
    with pytest.raises(InsufficientMaterial):
        retrieve_for_themes(index, KeywordEmbedder(), "Ricetta", random.Random(1))
