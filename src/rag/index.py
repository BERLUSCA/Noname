"""Chunk index and retrieval: cosine similarity over normalized embeddings.

The embedding model name is NOT in this file: it comes from the caller (CLI or EMBEDDING_MODEL in .env).
"""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.ingestion.chunking import MIN_CHUNK_CHARS, chunk_text, clean_text

MIN_TOP_K, MAX_TOP_K = 3, 5  # guardrail 6: top_k between 3 and 5


@dataclass(frozen=True)
class Chunk:
    id: str
    slug: str
    title: str
    url: str
    text: str


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def build_chunks(raw_dir: Path, max_chars: int = 600, overlap: int = 100) -> list[Chunk]:
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for doc in manifest:
        text = clean_text((raw_dir / f"{doc['slug']}.txt").read_text(encoding="utf-8"))
        for n, piece in enumerate(chunk_text(text, max_chars, overlap)):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(id=f"{doc['slug']}#{n}", slug=doc["slug"], title=doc["title"],
                                url=doc["url"], text=piece))
    return chunks


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("Numero di chunk e vettori diverso")
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorIndex":
        vectors = embedder.embed([c.text for c in chunks])
        return cls(chunks, _normalize(vectors))

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[float, Chunk]]:
        if not MIN_TOP_K <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k deve essere tra {MIN_TOP_K} e {MAX_TOP_K}")
        query = _normalize(query_vector.reshape(1, -1))[0]
        scores = self.vectors @ query
        order = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.chunks[i]) for i in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        vectors = np.load(directory / "vectors.npy")
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        return cls(chunks, vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def retrieve(index: VectorIndex, embedder: Embedder, query: str, top_k: int = 4) -> list[tuple[float, Chunk]]:
    """Returns the top_k chunks for a query, each with its similarity score and source."""
    query_vector = embedder.embed([query])[0]
    return index.search(query_vector, top_k=top_k)


class InsufficientMaterial(Exception):
    """Raised when the corpus has no document for the requested title (guardrail 3)."""


def normalize_title(title: str) -> str:
    """Lowercase, no accents, no punctuation, no parenthetical disambiguator: "Il padrino (film)" -> "il padrino"."""
    title = re.sub(r"\([^)]*\)", " ", title)
    ascii_text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.lower()))


# User-facing alternative titles -> title used in the corpus (both as typed, normalized on lookup).
TITLE_ALIASES: dict[str, str] = {"Psycho": "Psyco"}


def _title_indices(index: VectorIndex, title: str) -> list[int]:
    """Indices of the chunks whose document matches the title (aliases applied, prefix of whole words allowed)."""
    wanted = normalize_title(TITLE_ALIASES.get(title.strip(), title))
    matching = [
        i for i, c in enumerate(index.chunks)
        if normalize_title(c.title) == wanted or normalize_title(c.title).startswith(wanted + " ")
    ]
    if not matching:
        raise InsufficientMaterial(f"Nessun documento nel corpus per il titolo: {title}")
    return matching


def retrieve_for_title(index: VectorIndex, embedder: Embedder, title: str, top_k: int = 4) -> list[tuple[float, Chunk]]:
    """Best chunks of the document whose title matches (aliases applied). Raises InsufficientMaterial if none does."""
    matching = _title_indices(index, title)
    query = _normalize(embedder.embed([title]))[0]
    scores = index.vectors[matching] @ query
    order = np.argsort(-scores)[:top_k]
    return [(float(scores[j]), index.chunks[matching[j]]) for j in order]


# Topics a quiz can cover. Each request uses a random subset, so the passage changes between quizzes.
THEMES = [
    "trama",
    "regia e produzione",
    "cast e personaggi",
    "premi e riconoscimenti",
    "accoglienza critica",
    "uscita e incassi",
]


def retrieve_for_themes(index: VectorIndex, embedder: Embedder, title: str, rng,
                        themes_per_request: int = 2, per_theme: int = 2) -> list[tuple[float, Chunk]]:
    """Chunks of the title's document for a random subset of themes: varied passages, fewer repeats."""
    matching = _title_indices(index, title)
    chosen = rng.sample(THEMES, themes_per_request)
    picked: list[tuple[float, Chunk]] = []
    seen: set[int] = set()
    for theme in chosen:
        query = _normalize(embedder.embed([f"{title} {theme}"]))[0]
        scores = index.vectors[matching] @ query
        taken = 0
        for j in np.argsort(-scores):
            idx = matching[int(j)]
            if idx in seen:
                continue
            seen.add(idx)
            picked.append((float(scores[j]), index.chunks[idx]))
            taken += 1
            if taken == per_theme:
                break
    return picked


def best_similar_chunk(index: VectorIndex, embedder: Embedder, fact: str, title: str) -> tuple[float, Chunk]:
    """Chunk of the title's document most similar to the claim, searched over the WHOLE document.

    Similarity measures topic, not truth: use it to pick the text to verify, never as a truth check.
    """
    wanted = normalize_title(TITLE_ALIASES.get(title.strip(), title))
    matching = [
        i for i, c in enumerate(index.chunks)
        if normalize_title(c.title) == wanted or normalize_title(c.title).startswith(wanted + " ")
    ]
    if not matching:
        raise InsufficientMaterial(f"Nessun documento nel corpus per il titolo: {title}")
    fact_vector = _normalize(embedder.embed([fact]))[0]
    scores = index.vectors[matching] @ fact_vector
    best = int(np.argmax(scores))
    return float(scores[best]), index.chunks[matching[best]]

