"""
Ex. No: 1
TEXT GENERATION USING PRE-TRAINED FOUNDATION MODELS
"""
from pathlib import Path

from transformers import pipeline, set_seed

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "input" / "prompt.txt"
OUTPUT_FILE = ROOT / "output" / "generated_text.txt"


def main() -> None:
    prompt = INPUT_FILE.read_text(encoding="utf-8").strip()
    print("Loading GPT-2 text-generation pipeline...")
    generator = pipeline("text-generation", model="gpt2")
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    set_seed(42)

    outputs = generator(
        prompt,
        max_new_tokens=50,
        num_return_sequences=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    lines = [
        "Ex. No: 1 - Text Generation Using Pre-Trained Foundation Models",
        f"Model: gpt2",
        f"Prompt: {prompt}",
        "",
    ]
    for i, out in enumerate(outputs, 1):
        header = f"--- Generated Text {i} ---"
        print(header)
        print(out["generated_text"])
        print()
        lines.append(header)
        lines.append(out["generated_text"])
        lines.append("")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
