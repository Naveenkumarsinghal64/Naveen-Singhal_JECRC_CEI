"""
chunking.py
-----------
Stage 2: Text Chunking.

Splits a large block of raw text into smaller overlapping chunks.
Smaller chunks improve retrieval accuracy because embeddings capture
the meaning of a focused piece of text better than a huge one.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters.

    Args:
        text: The raw document text.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between
                       consecutive chunks (helps preserve context
                       that spans a chunk boundary).

    Returns:
        A list of text chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Normalize whitespace so chunks aren't full of stray newlines/spaces.
    cleaned = " ".join(text.split())

    if len(cleaned) <= chunk_size:
        return [cleaned] if cleaned else []

    chunks = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = start + chunk_size
        chunk = cleaned[start:end]
        chunks.append(chunk.strip())

        if end >= text_length:
            break

        start = end - chunk_overlap  # slide window back to keep overlap

    return [c for c in chunks if c]
