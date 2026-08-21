"""
Ex. No: 5
SENTIMENT ANALYSIS AND DOCUMENT CLASSIFICATION USING FOUNDATION MODELS
"""
from pathlib import Path

from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
REVIEWS_FILE = ROOT / "input" / "reviews.txt"
DOCUMENT_FILE = ROOT / "input" / "document.txt"
LABELS_FILE = ROOT / "input" / "candidate_labels.txt"
OUTPUT_FILE = ROOT / "output" / "classification_results.txt"

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "typeform/distilbert-base-uncased-mnli"


def main() -> None:
    reviews = [line.strip() for line in REVIEWS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    document = DOCUMENT_FILE.read_text(encoding="utf-8").strip()
    candidate_labels = [
        line.strip()
        for line in LABELS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"Loading sentiment model: {SENTIMENT_MODEL}")
    sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

    report = [
        "Ex. No: 5 - Sentiment Analysis and Document Classification",
        f"Sentiment model: {SENTIMENT_MODEL}",
        f"Zero-shot model: {ZERO_SHOT_MODEL}",
        "",
        "--- Sentiment Analysis ---",
        "",
    ]

    for review in reviews:
        result = sentiment_analyzer(review)[0]
        line = f"Review: {review}\n -> {result['label']} ({round(result['score'], 3)})\n"
        print(line)
        report.append(line)

    print(f"Loading zero-shot classification model: {ZERO_SHOT_MODEL}")
    classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    classification = classifier(document, candidate_labels)

    print("Document:", document)
    report.append("--- Document Classification (Zero-Shot) ---")
    report.append(f"Document: {document}")
    for label, score in zip(classification["labels"], classification["scores"]):
        row = f"{label}: {round(score, 3)}"
        print(row)
        report.append(row)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"\nOutput written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
