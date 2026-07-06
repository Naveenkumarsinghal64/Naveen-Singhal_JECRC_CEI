"""
embeddings.py
-------------
Stage 3: Embedding Creation.

Converts text (chunks or a user query) into dense vector representations
using a free, local sentence-transformers model. No API key required.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        all-MiniLM-L6-v2 is small, fast, runs on CPU, and gives solid
        quality for semantic similarity search — a good beginner choice.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text chunks. Returns a (N, dim) float32 array."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # so cosine similarity == dot product
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a (1, dim) float32 array."""
        return self.embed_texts([query])
