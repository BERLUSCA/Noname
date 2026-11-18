"""Splits a document into overlapping chunks, keeping paragraphs together when possible."""


def chunk_text(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
    """Packs paragraphs into chunks of at most max_chars. Long paragraphs are split with overlap."""
    if max_chars <= overlap:
        raise ValueError("max_chars deve essere maggiore di overlap")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long(paragraph, max_chars, overlap))
            continue
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max_chars - overlap
    return [text[start:start + max_chars] for start in range(0, len(text), step) if text[start:start + max_chars].strip()]


# Wikipedia boilerplate that is not quiz material: section headings and navigation sections.
BOILERPLATE_LINES = {"Note", "Voci correlate", "Altri progetti", "Riconoscimenti", "Bibliografia", "Collegamenti esterni"}
MIN_CHUNK_CHARS = 80  # shorter pieces are fragments, not information


def clean_text(text: str) -> str:
    """Drops external-link lists, section headings and navigation boilerplate: noise for retrieval."""
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("(EN") or ", su " in line:
            continue
        if stripped.startswith("==") or stripped.strip("= ") in BOILERPLATE_LINES:
            continue
        kept.append(line)
    return "\n".join(kept)
