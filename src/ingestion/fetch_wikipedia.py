"""Downloads the fixed Wikipedia corpus (plain text) into data/raw/ with a manifest.

Source: it.wikipedia.org, text under CC BY-SA. Run: python -m src.ingestion.fetch_wikipedia
"""

import hashlib
import json
import time
import urllib.error
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
API_URL = "https://it.wikipedia.org/w/api.php"
USER_AGENT = "QuantoneSai-student-project/0.1 (educational; contact: local)"

# Fixed corpus: the works used in evals/gold.jsonl, so retrieval can be checked against them.
CORPUS_TITLES = [
    "Inception",
    "Il padrino (film)",
    "Breaking Bad",
    "Friends",
    "Pulp Fiction",
    "Il Signore degli Anelli - La Compagnia dell'Anello",
    "Psycho",
    "Il miglio verde (film)",
    "Obsession (film 2025)",
    "Il silenzio degli innocenti (film)",
    "Matrix",
    "Forrest Gump",
    "Fight Club (film)",
    "Interstellar",
    "Il gladiatore",
    "Titanic (film 1997)",
    "Il cavaliere oscuro",
    "La vita è bella",
    "Parasite (film 2019)",
]


def slugify(title: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def _get_json(request: urllib.request.Request, retries: int = 4) -> dict:
    """GET with backoff on HTTP 429 (rate limit)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_extract(title: str) -> dict:
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": title, "format": "json", "formatversion": 2,
    }
    request = urllib.request.Request(
        f"{API_URL}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    data = _get_json(request)
    page = data["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        raise ValueError(f"Pagina non trovata o vuota: {title}")
    return {
        "title": page["title"],
        "url": "https://it.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
        "text": page["extract"],
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for title in CORPUS_TITLES:
        time.sleep(2)  # be polite to the API
        page = fetch_extract(title)
        slug = slugify(page["title"])
        (RAW_DIR / f"{slug}.txt").write_text(page["text"], encoding="utf-8")
        manifest.append({"slug": slug, "title": page["title"], "url": page["url"],
                         "retrieved_at": retrieved_at, "chars": len(page["text"]),
                         "sha256": hashlib.sha256((RAW_DIR / f"{slug}.txt").read_bytes()).hexdigest()})
        print(f"OK {page['title']} ({len(page['text'])} caratteri)")
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {len(manifest)} pagine")


if __name__ == "__main__":
    main()
