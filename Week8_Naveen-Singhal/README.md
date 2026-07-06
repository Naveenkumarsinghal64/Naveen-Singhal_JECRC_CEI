# Document Question Answering System (RAG)

A simple **Retrieval-Augmented Generation (RAG)** system that answers
questions from custom documents (PDF or text). Instead of relying only
on a language model's internal knowledge, the system retrieves relevant
chunks of text from your document and generates an answer grounded in
that content — fully local, no API key required.

## Overview

This project implements a RAG pipeline: it retrieves relevant
information from a document and then uses a language model to generate
an answer, improving factual accuracy for private/domain-specific data
(resumes, notes, research papers, books, etc).

## Objectives

- Understand the concept of Retrieval-Augmented Generation (RAG)
- Build a pipeline combining retrieval and generation
- Enable question answering over custom documents (PDF / text)
- Learn how modern RAG-based AI systems work internally

## System Architecture

```
PDF/TXT file
     |
     v
[1] Document Ingestion   -> raw text extracted (src/document_loader.py)
     |
     v
[2] Text Chunking        -> text split into overlapping chunks (src/chunking.py)
     |
     v
[3] Embedding Creation    -> each chunk converted to a vector (src/embeddings.py)
     |
     v
[4] Vector Database       -> vectors stored in FAISS (src/vector_store.py)
     |
     v
User Question
     |
     v
[5] Query Embedding       -> question converted to a vector (src/embeddings.py)
     |
     v
[6] Context Retrieval     -> most similar chunks fetched (src/vector_store.py)
     |
     v
[7] Answer Generation     -> LLM generates grounded answer (src/generator.py)
     |
     v
Final Answer
```

## Tech Stack (100% free, runs locally)

| Component        | Tool                              |
|-------------------|-----------------------------------|
| Document loading   | `pypdf`                          |
| Embeddings         | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store       | `faiss-cpu`                      |
| Answer generation   | `transformers` (`google/flan-t5-base`) |

No OpenAI/Anthropic API key needed — everything runs on your own machine (CPU is fine).

## Project Structure

```
rag-document-qa/
├── README.md
├── requirements.txt
├── main.py                  # CLI entry point
├── documents/                # put your PDF/txt files here
└── src/
    ├── document_loader.py    # Stage 1: load PDF/txt
    ├── chunking.py           # Stage 2: split text into chunks
    ├── embeddings.py         # Stage 3 & 5: text -> vectors
    ├── vector_store.py       # Stage 4 & 6: FAISS store + retrieval
    ├── generator.py          # Stage 7: LLM answer generation
    └── rag_pipeline.py       # Combines all stages together
```

## Setup

1. Clone the repo and move into it:
   ```bash
   git clone <your-repo-url>
   cd rag-document-qa
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your PDF or `.txt` file inside the `documents/` folder
   (e.g. your resume, notes, or a research paper).

2. Run the pipeline:
   ```bash
   python main.py --file documents/your_file.pdf
   ```

3. On first run, the embedding model and `flan-t5-base` will download
   automatically from Hugging Face (a few hundred MB — needs internet
   only for this first download).

4. Once indexing is done, ask questions in the terminal:
   ```
   Your question: What is the main idea of this document?

   Answer: ...

   --- Retrieved context (for reference) ---
   [1] (score: 0.812) ...chunk preview...
   [2] (score: 0.774) ...chunk preview...
   ```

5. Type `exit` or `quit` to stop.

## Example Flow

**User Question:** "What is the main idea of the document?"

**System Process:**
1. Question is converted into an embedding
2. Most similar chunks are retrieved from the FAISS index
3. Retrieved chunks are passed as context to the LLM
4. LLM generates a concise, grounded answer

## Key Concepts

1. **Retrieval** — finds the most relevant chunks of text using
   embeddings and vector similarity search.
2. **Augmentation** — retrieved content is added to the model's input
   as context.
3. **Generation** — the language model produces the final answer,
   grounded in the retrieved context (reduces hallucination).

## Possible Improvements

- Better chunking strategies (semantic chunking instead of fixed-size)
- Try different/larger embedding models
- Hybrid search (keyword + vector search)
- Add a re-ranking step for better relevance
- Swap in a larger LLM (local or via API) for higher quality answers
- Add a simple web UI (e.g. Streamlit/Gradio) on top of `RAGPipeline`

## Key Learnings

- How RAG systems combine retrieval and generation
- Importance of retrieval quality in overall answer accuracy
- Working with embeddings and vector databases
- Handling unstructured text data (PDFs)
- Designing a modular, scalable AI pipeline

## Conclusion

This project demonstrates how to build a system that understands user
queries, retrieves relevant information from custom documents, and
generates accurate, grounded answers. RAG systems like this are widely
used in chatbots, knowledge assistants, enterprise search, and
AI-powered documentation tools.
