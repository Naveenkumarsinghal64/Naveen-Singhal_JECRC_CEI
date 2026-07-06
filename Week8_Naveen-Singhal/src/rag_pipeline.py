"""
rag_pipeline.py
---------------
Ties together every stage of the Retrieval-Augmented Generation system:

Document Ingestion -> Chunking -> Embedding -> Vector Store
  -> Query Embedding -> Retrieval -> Generation
"""

from typing import List, Tuple

from src.document_loader import load_document
from src.chunking import chunk_text
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.generator import AnswerGenerator


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print("Loading embedding model...")
        self.embedder = EmbeddingModel(embedding_model_name)

        print("Loading language model (first run downloads it, please wait)...")
        self.generator = AnswerGenerator(llm_model_name)

        self.vector_store: VectorStore = None

    def ingest(self, file_path: str) -> int:
        """
        Run the ingestion side of the pipeline on a document:
        load -> chunk -> embed -> store.

        Returns the number of chunks created.
        """
        print(f"Loading document: {file_path}")
        raw_text = load_document(file_path)

        print("Splitting into chunks...")
        chunks = chunk_text(raw_text, self.chunk_size, self.chunk_overlap)
        print(f"Created {len(chunks)} chunks.")

        print("Creating embeddings...")
        embeddings = self.embedder.embed_texts(chunks)

        self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        self.vector_store.add(embeddings, chunks)

        print("Document indexed successfully.\n")
        return len(chunks)

    def query(self, question: str, top_k: int = 3) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Run the query side of the pipeline:
        embed query -> retrieve chunks -> generate answer.

        Returns (answer, retrieved_chunks_with_scores).
        """
        if self.vector_store is None:
            raise RuntimeError("No document has been ingested yet. Call ingest() first.")

        query_embedding = self.embedder.embed_query(question)
        retrieved = self.vector_store.search(query_embedding, top_k=top_k)

        if not retrieved:
            return "No relevant content found in the document.", []

        context_chunks = [chunk for chunk, _score in retrieved]
        answer = self.generator.generate_answer(question, context_chunks)

        return answer, retrieved
