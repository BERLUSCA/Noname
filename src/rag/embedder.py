"""Local embeddings with sentence-transformers (no API cost). The model name is passed in, never hardcoded."""

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False))
