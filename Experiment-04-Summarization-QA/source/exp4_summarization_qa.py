"""
Ex. No: 4
TEXT SUMMARIZATION AND QUESTION-ANSWERING SYSTEM USING LARGE LANGUAGE MODELS
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTICLE_FILE = ROOT / "input" / "article.txt"
QUESTION_FILE = ROOT / "input" / "question.txt"
OUTPUT_FILE = ROOT / "output" / "summarization_qa_results.txt"

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-6-6"
QA_MODEL = "distilbert-base-cased-distilled-squad"


def summarize(article: str) -> str:
    print(f"Loading summarization model: {SUMMARIZER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        **inputs,
        max_length=45,
        min_length=20,
        num_beams=4,
        do_sample=False,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def answer_question(question: str, context: str) -> tuple[str, float]:
    print(f"Loading question-answering model: {QA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    start_idx = int(torch.argmax(outputs.start_logits))
    end_idx = int(torch.argmax(outputs.end_logits))
    if end_idx < start_idx:
        end_idx = start_idx
    answer = tokenizer.decode(inputs["input_ids"][0][start_idx : end_idx + 1], skip_special_tokens=True)
    start_p = F.softmax(outputs.start_logits, dim=-1)[0, start_idx]
    end_p = F.softmax(outputs.end_logits, dim=-1)[0, end_idx]
    score = float((start_p * end_p).item())
    return answer, score


def main() -> None:
    article = ARTICLE_FILE.read_text(encoding="utf-8").strip()
    question = QUESTION_FILE.read_text(encoding="utf-8").strip()

    summary_text = summarize(article)
    print("Summary:")
    print(summary_text)

    answer, score = answer_question(question, article)
    print("\nQuestion:", question)
    print("Answer:", answer, "| Confidence:", round(score, 3))

    report = [
        "Ex. No: 4 - Text Summarization and Question Answering",
        f"Summarization model: {SUMMARIZER_MODEL}",
        f"QA model: {QA_MODEL}",
        "",
        "ARTICLE:",
        article,
        "",
        "Summary:",
        summary_text,
        "",
        f"Question: {question}",
        f"Answer: {answer} | Confidence: {round(score, 3)}",
    ]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"\nOutput written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
