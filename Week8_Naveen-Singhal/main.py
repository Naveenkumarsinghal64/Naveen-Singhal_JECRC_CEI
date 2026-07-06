"""
main.py
-------
Command-line entry point for the Document Question Answering (RAG) system.

Usage:
    python main.py --file documents/resume.pdf

Then type your questions at the prompt. Type 'exit' or 'quit' to stop.
"""

import argparse
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about a document using a local RAG pipeline."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the PDF or .txt file to load (e.g. documents/resume.pdf)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question (default: 3)",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.ingest(args.file)

    print("Document ready! Ask questions about it below.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ("exit", "quit", ""):
            print("Goodbye!")
            break

        answer, retrieved_chunks = pipeline.query(question, top_k=args.top_k)

        print(f"\nAnswer: {answer}\n")
        print("--- Retrieved context (for reference) ---")
        for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
            preview = chunk[:150] + ("..." if len(chunk) > 150 else "")
            print(f"[{i}] (score: {score:.3f}) {preview}")
        print()


if __name__ == "__main__":
    main()
