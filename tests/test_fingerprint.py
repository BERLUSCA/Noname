"""Tests for corpus and index fingerprints: temporary folders, no network."""

import json

import pytest

from src.rag.fingerprint import (
    IndexMismatch,
    build_fingerprint,
    check_corpus,
    corpus_hashes,
    file_sha256,
    save_fingerprint,
    verify_index,
)


def make_corpus(tmp_path, text="Testo del film, abbastanza lungo per essere un documento reale."):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "inception.txt").write_text(text, encoding="utf-8")
    manifest = [{"slug": "inception", "title": "Inception", "url": "u",
                 "sha256": file_sha256(raw / "inception.txt")}]
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw


def test_consistent_corpus_has_no_anomalies(tmp_path):
    assert check_corpus(make_corpus(tmp_path)) == []


def test_modified_file_is_reported(tmp_path):
    raw = make_corpus(tmp_path)
    (raw / "inception.txt").write_text("testo alterato", encoding="utf-8")
    assert any("modificato" in p for p in check_corpus(raw))


def test_missing_file_is_reported(tmp_path):
    raw = make_corpus(tmp_path)
    (raw / "inception.txt").unlink()
    assert any("mancante" in p for p in check_corpus(raw))


def test_missing_hash_in_manifest_is_reported(tmp_path):
    raw = make_corpus(tmp_path)
    (raw / "manifest.json").write_text(json.dumps([{"slug": "inception", "title": "I", "url": "u"}]), encoding="utf-8")
    assert any("hash mancante" in p for p in check_corpus(raw))


def test_index_matches_when_nothing_changed(tmp_path):
    raw = make_corpus(tmp_path)
    index_dir = tmp_path / "index"
    save_fingerprint(build_fingerprint(raw, "modello", 600, 100), index_dir)
    verify_index(index_dir, build_fingerprint(raw, "modello", 600, 100))  # no exception


def test_index_mismatch_when_corpus_changed(tmp_path):
    raw = make_corpus(tmp_path)
    index_dir = tmp_path / "index"
    save_fingerprint(build_fingerprint(raw, "modello", 600, 100), index_dir)
    (raw / "inception.txt").write_text("altro testo", encoding="utf-8")
    with pytest.raises(IndexMismatch, match="Corpus cambiato"):
        verify_index(index_dir, build_fingerprint(raw, "modello", 600, 100))


def test_index_mismatch_when_model_changed(tmp_path):
    raw = make_corpus(tmp_path)
    index_dir = tmp_path / "index"
    save_fingerprint(build_fingerprint(raw, "vecchio", 600, 100), index_dir)
    with pytest.raises(IndexMismatch, match="altro modello"):
        verify_index(index_dir, build_fingerprint(raw, "nuovo", 600, 100))


def test_index_mismatch_when_chunking_changed(tmp_path):
    raw = make_corpus(tmp_path)
    index_dir = tmp_path / "index"
    save_fingerprint(build_fingerprint(raw, "modello", 600, 100), index_dir)
    with pytest.raises(IndexMismatch, match="chunking"):
        verify_index(index_dir, build_fingerprint(raw, "modello", 400, 100))


def test_index_without_fingerprint_is_mismatch(tmp_path):
    raw = make_corpus(tmp_path)
    with pytest.raises(IndexMismatch, match="senza impronta"):
        verify_index(tmp_path / "index", build_fingerprint(raw, "modello", 600, 100))


def test_corpus_hashes_are_keyed_by_slug(tmp_path):
    assert set(corpus_hashes(make_corpus(tmp_path))) == {"inception"}


def test_api_returns_503_with_reason_when_index_does_not_match(tmp_path):
    from fastapi.testclient import TestClient

    from src.api.db import Database
    from src.api.main import create_app
    from tests.fakes import FakeVerifier
    from tests.test_api import FakeGenerator

    def broken_retriever():
        raise IndexMismatch("Corpus cambiato rispetto all'indice (inception): ricostruire")

    app = create_app(Database(":memory:"), lambda: FakeGenerator(), lambda: broken_retriever(), lambda: FakeVerifier())
    response = TestClient(app).post("/quiz", json={"title": "Inception", "n_questions": 1})
    assert response.status_code == 503
    assert "ricostruire" in response.json()["detail"]
