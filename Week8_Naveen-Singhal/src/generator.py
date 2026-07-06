"""
generator.py
------------
Stage 7: Answer Generation.

Uses a free, local Hugging Face model (google/flan-t5-base) to generate
an answer grounded in the retrieved context. No API key or paid API
required — the model runs directly on your machine (CPU is fine).
"""

from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AnswerGenerator:
    """Wrapper around a local HF seq2seq (T5-family) model, loaded directly."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)

        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say "
            "\"I could not find this in the document.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()