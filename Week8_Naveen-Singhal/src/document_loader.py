"""
document_loader.py
-------------------
Stage 1: Document Ingestion.

Loads a document (PDF or plain .txt) from disk and converts it into
a single raw text string that can be passed on to the chunking stage.
"""

import os
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file, page by page."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(file_path)
    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)

    if not full_text.strip():
        raise ValueError(
            "No extractable text found in the PDF. "
            "It might be a scanned/image-only PDF (would need OCR)."
        )

    return full_text


def load_txt(file_path: str) -> str:
    """Read raw text from a plain .txt file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(file_path: str) -> str:
    """
    Auto-detect the file type from its extension and load the text.
    Supports: .pdf, .txt
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")
