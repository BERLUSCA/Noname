"""Corpus and index fingerprints: detect edited corpus files and indexes built from another corpus or model.

This detects mistakes and drift. It is not a security control: anyone who can write data/ can recompute hashes.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class IndexMismatch(Exception):
    """The index does not match the current corpus, model or chunking parameters."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_hashes(raw_dir: Path) -> dict[str, str]:
    """SHA-256 of every document listed in the manifest, keyed by slug."""
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    return {doc["slug"]: file_sha256(raw_dir / f"{doc['slug']}.txt") for doc in manifest}


def check_corpus(raw_dir: Path) -> list[str]:
    """Anomalies in the corpus against the hashes stored in the manifest. Empty list = consistent."""
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for doc in manifest:
        path = raw_dir / f"{doc['slug']}.txt"
        if not path.exists():
            problems.append(f"file mancante: {doc['slug']}.txt")
            continue
        expected = doc.get("sha256")
        if expected is None:
            problems.append(f"hash mancante nel manifest: {doc['slug']}")
        elif file_sha256(path) != expected:
            problems.append(f"file modificato: {doc['slug']}.txt")
    return problems


@dataclass(frozen=True)
class IndexFingerprint:
    corpus: dict[str, str]
    model: str
    max_chars: int
    overlap: int

    def digest(self) -> str:
        payload = json.dumps({"corpus": self.corpus, "model": self.model,
                              "max_chars": self.max_chars, "overlap": self.overlap}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fingerprint(raw_dir: Path, model: str, max_chars: int, overlap: int) -> IndexFingerprint:
    return IndexFingerprint(corpus=corpus_hashes(raw_dir), model=model, max_chars=max_chars, overlap=overlap)


def save_fingerprint(fp: IndexFingerprint, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "fingerprint.json").write_text(
        json.dumps({"digest": fp.digest(), "corpus": fp.corpus, "model": fp.model,
                    "max_chars": fp.max_chars, "overlap": fp.overlap}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def verify_index(index_dir: Path, current: IndexFingerprint) -> None:
    """Raises IndexMismatch with the reason when the index was built from something else."""
    path = index_dir / "fingerprint.json"
    if not path.exists():
        raise IndexMismatch("Indice senza impronta: ricostruire con src.rag.build_index")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["model"] != current.model:
        raise IndexMismatch(f"Indice costruito con un altro modello ({saved['model']}): ricostruire")
    if (saved["max_chars"], saved["overlap"]) != (current.max_chars, current.overlap):
        raise IndexMismatch("Parametri di chunking cambiati: ricostruire l'indice")
    if saved["corpus"] != current.corpus:
        changed = sorted(set(saved["corpus"]) ^ set(current.corpus)
                         | {s for s in saved["corpus"] if s in current.corpus and saved["corpus"][s] != current.corpus[s]})
        raise IndexMismatch(f"Corpus cambiato rispetto all'indice ({', '.join(changed)}): ricostruire")


def _cli() -> None:
    """python -m src.rag.fingerprint check    -> reports corpus anomalies
    python -m src.rag.fingerprint stamp    -> writes sha256 for the files currently on disk (explicit, trust on first use)
    """
    import sys

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "stamp":
        manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
        for doc in manifest:
            doc["sha256"] = file_sha256(raw_dir / f"{doc['slug']}.txt")
        (raw_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Impronte scritte per {len(manifest)} documenti")
        return
    problems = check_corpus(raw_dir)
    print("Corpus coerente" if not problems else "\n".join(problems))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
