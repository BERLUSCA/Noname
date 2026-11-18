"""Fakes for API tests: one chunk for any title, nothing for titles containing 'Ricetta'; a judge that
rejects questions containing 'BAD'."""

from src.rag.index import Chunk, InsufficientMaterial


class FakeRetriever:
    def retrieve(self, title: str) -> list[Chunk]:
        if "Ricetta" in title:
            raise InsufficientMaterial(f"Nessun documento nel corpus per il titolo: {title}")
        return [Chunk(id="fake#0", slug="fake", title=title, url="https://example.org/fake", text="Testo di prova.")]


class FakeVerifier:
    """Judge fake: a claim is supported unless it contains 'BAD' (marks a question to reject)."""

    def __init__(self):
        self.calls = 0

    def verify_many(self, claims, text):
        self.calls += 1
        return ["BAD" not in claim for claim in claims]
