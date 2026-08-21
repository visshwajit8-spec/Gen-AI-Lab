"""
Ex. No: 6
RETRIEVAL-AUGMENTED GENERATION (RAG) SYSTEM USING VECTOR DATABASES
"""
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
KB_FILE = ROOT / "input" / "knowledge_base.txt"
QUERY_FILE = ROOT / "input" / "query.txt"
OUTPUT_FILE = ROOT / "output" / "rag_results.txt"

EMBED_MODEL = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-small"


def generate_answer(prompt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    documents = [line.strip() for line in KB_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    query = QUERY_FILE.read_text(encoding="utf-8").strip()

    print(f"Embedding documents with {EMBED_MODEL}...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    doc_embeddings = embed_model.encode(documents, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings)

    query_embedding = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _distances, indices = index.search(query_embedding, k=2)
    retrieved_chunks = [documents[i] for i in indices[0]]

    context = " ".join(retrieved_chunks)
    prompt = (
        "Use the context to answer the question in one sentence.\n"
        f"Context: {context}\nQuestion: {query}\nAnswer:"
    )
    print("Vector database: FAISS IndexFlatIP (cosine similarity)")

    print(f"Generating grounded answer with {GENERATOR_MODEL}...")
    answer = generate_answer(prompt)

    print("Retrieved Context:", retrieved_chunks)
    print("Answer:", answer)

    report = [
        "Ex. No: 6 - Retrieval-Augmented Generation (RAG)",
        f"Embedding model: {EMBED_MODEL}",
        "Vector database: FAISS IndexFlatIP (cosine similarity)",
        f"Generator: {GENERATOR_MODEL}",
        "",
        f"Query: {query}",
        "",
        "Retrieved Context:",
        *[f"- {chunk}" for chunk in retrieved_chunks],
        "",
        f"Answer: {answer}",
    ]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
