"""
Ex. No: 2
PROMPT ENGINEERING TECHNIQUES FOR CONTENT GENERATION,
REASONING AND TASK AUTOMATION
"""
from pathlib import Path

from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_FILE = ROOT / "output" / "prompt_engineering_results.txt"


def main() -> None:
    zero_shot_prompt = (INPUT_DIR / "zero_shot_prompt.txt").read_text(encoding="utf-8").strip()
    few_shot_prompt = (INPUT_DIR / "few_shot_prompt.txt").read_text(encoding="utf-8").strip()
    cot_prompt = (INPUT_DIR / "chain_of_thought_prompt.txt").read_text(encoding="utf-8").strip()

    print("Loading GPT-2 pipeline for prompt-engineering comparison...")
    generator = pipeline("text-generation", model="gpt2")
    generator.tokenizer.pad_token = generator.tokenizer.eos_token

    prompts = [
        ("Zero-shot", zero_shot_prompt),
        ("Few-shot", few_shot_prompt),
        ("Chain-of-Thought", cot_prompt),
    ]

    report = [
        "Ex. No: 2 - Prompt Engineering Techniques",
        "Model: gpt2",
        "",
    ]

    for name, prompt in prompts:
        out = generator(
            prompt,
            max_new_tokens=40,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        generated = out[0]["generated_text"]
        print(f"=== {name} ===")
        print(generated)
        print()
        report.append(f"=== {name} ===")
        report.append(f"PROMPT:\n{prompt}")
        report.append("")
        report.append(f"OUTPUT:\n{generated}")
        report.append("")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
