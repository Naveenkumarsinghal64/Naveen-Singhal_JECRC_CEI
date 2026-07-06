"""
vector_store.py
----------------
Stage 4 & 6: Vector Database + Context Retrieval.

Stores chunk embeddings in a FAISS index and retrieves the most
similar chunks for a given query embedding.
"""

from typing import List, Tuple
import numpy as np
import faiss


class VectorStore:
    """A simple in-memory FAISS vector store."""

    def __init__(self, embedding_dim: int):
        # Since embeddings are normalized, inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[str] = []

    def add(self, embeddings: np.ndarray, chunks: List[str]) -> None:
        """Add chunk embeddings and their corresponding text to the store."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find the top_k most similar chunks to the query embedding.

        Returns a list of (chunk_text, similarity_score) tuples,
        ordered from most to least relevant.
        """
        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))

        return results
