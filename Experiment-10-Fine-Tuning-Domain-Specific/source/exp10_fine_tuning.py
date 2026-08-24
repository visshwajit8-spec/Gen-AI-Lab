"""
Ex. No: 10
FINE-TUNING A PRE-TRAINED LANGUAGE MODEL FOR A DOMAIN-SPECIFIC APPLICATION
"""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "input" / "reviews_dataset.json"
OUTPUT_FILE = ROOT / "output" / "fine_tuning_results.txt"
OUTPUT_TXT = ROOT / "output" / "output.txt"
SAVE_DIR = ROOT / "output" / "fine_tuned_distilbert_imdb"

MODEL_NAME = "distilbert-base-uncased"


class TextClassificationDataset(Dataset):
    def __init__(self, items, tokenizer, max_length=128):
        self.texts = [x["text"] for x in items]
        self.labels = [x["label"] for x in items]
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_model_and_tokenizer():
    for name in [MODEL_NAME, "distilbert-base-uncased-finetuned-sst-2-english"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, local_files_only=True, ignore_mismatched_sizes=True)
            print(f"Loaded {name} from local cache.")
            return tokenizer, model
        except Exception:
            continue
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2, ignore_mismatched_sizes=True)
    return tokenizer, model


def main() -> None:
    print(f"Loading dataset from {DATASET_FILE}...")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = data["train"]
    test_data = data["test"]
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer, model = load_model_and_tokenizer()


    train_dataset = TextClassificationDataset(train_data, tokenizer)
    test_dataset = TextClassificationDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 2

    print("\n--- Starting Fine-Tuning ---")
    training_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = round(total_loss / len(train_loader), 2)
        epoch_acc = round(correct / total, 2)
        log_line = f"Epoch {epoch}/{epochs} - loss: {epoch_loss} - accuracy: {epoch_acc}"
        print(log_line)
        training_log.append(log_line)

    # Evaluation
    print("\n--- Evaluating Model ---")
    model.eval()
    eval_loss_total = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss_total += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_loss = round(eval_loss_total / len(test_loader), 2)
    eval_accuracy = round(accuracy_score(all_labels, all_preds), 3)
    eval_metrics = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    print("Evaluation metrics:", eval_metrics)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Fine-tuned model weights saved to {SAVE_DIR}")

    report = [
        "Ex. No: 10 - Fine-Tuning a Pre-Trained Language Model for Domain Classification",
        f"Base Model: {MODEL_NAME}",
        f"Training samples: {len(train_data)} | Test samples: {len(test_data)}",
        f"Number of Epochs: {epochs}",
        "",
        "--- Training Logs ---",
        *training_log,
        "",
        "--- Evaluation Results ---",
        f"Evaluation metrics: {eval_metrics}",
        f"Checkpoint Saved: {SAVE_DIR.name}",
    ]

    report_text = "\n".join(report)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report_text, encoding="utf-8")
    OUTPUT_TXT.write_text(report_text, encoding="utf-8")
    print(f"Output written to {OUTPUT_FILE} and {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
