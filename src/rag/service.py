"""Retrieval service used by the API: title in, source chunks out. Guardrail 3 lives here."""

import os
import random
from pathlib import Path

from dotenv import load_dotenv

from src.rag.build_index import MAX_CHARS, OVERLAP
from src.rag.embedder import SentenceEmbedder
from src.rag.fingerprint import IndexMismatch, build_fingerprint, verify_index
from src.rag.index import Chunk, InsufficientMaterial, VectorIndex, retrieve_for_themes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = ROOT / "data" / "index"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
MIN_TOP_K, MAX_TOP_K = 3, 5  # guardrail 6


class TitleRetriever:
    def __init__(self, index: VectorIndex, embedder, top_k: int = 4, rng: random.Random | None = None):
        if not MIN_TOP_K <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k deve essere tra {MIN_TOP_K} e {MAX_TOP_K}")
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.rng = rng or random.Random()

    def retrieve(self, title: str) -> list[Chunk]:
        """Chunks for a random subset of themes of the title. Raises InsufficientMaterial when none match."""
        picked = retrieve_for_themes(self.index, self.embedder, title, self.rng)
        return [chunk for _, chunk in picked][: self.top_k]


def load_retriever(model_name: str, index_dir: Path = DEFAULT_INDEX_DIR, raw_dir: Path = DEFAULT_RAW_DIR) -> TitleRetriever:
    """Loads the index only if it was built from the current corpus, model and chunking. Raises IndexMismatch."""
    verify_index(index_dir, build_fingerprint(raw_dir, model_name, MAX_CHARS, OVERLAP))
    return TitleRetriever(VectorIndex.load(index_dir), SentenceEmbedder(model_name))


def load_retriever_from_env() -> TitleRetriever:
    load_dotenv(ROOT / ".env")
    model_name = os.environ.get("EMBEDDING_MODEL", "")
    if not model_name:
        raise RuntimeError("Variabile EMBEDDING_MODEL non impostata in .env")
    return load_retriever(model_name)


__all__ = ["TitleRetriever", "InsufficientMaterial", "IndexMismatch", "load_retriever_from_env"]
