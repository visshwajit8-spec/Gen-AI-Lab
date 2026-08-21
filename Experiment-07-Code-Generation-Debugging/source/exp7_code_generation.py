"""
Ex. No: 7
AI-POWERED CODE GENERATION AND DEBUGGING ASSISTANT
"""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_FILE = ROOT / "output" / "code_generation_results.txt"
PROMPT1_FILE = INPUT_DIR / "prompt_prime.txt"
BUGGY_FILE = INPUT_DIR / "buggy_code.txt"

MODEL_NAME = "Salesforce/codegen-350M-mono"


def load_model():
    print(f"Loading code-generation model: {MODEL_NAME}...")
    tokenizer, model = None, None
    for name in ["Salesforce/codegen-350M-mono", "gpt2"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
            print(f"Loaded {name} from local cache.")
            break
        except Exception:
            continue

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model





def generate_code(tokenizer, model, prompt: str, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    prompt1 = PROMPT1_FILE.read_text(encoding="utf-8").strip()
    buggy_code = BUGGY_FILE.read_text(encoding="utf-8").strip()

    tokenizer, model = load_model()

    print("\n--- 1. Code Generation (Prime Checker) ---")
    gen_code_1 = generate_code(tokenizer, model, prompt1, max_new_tokens=80)
    print(gen_code_1)

    print("\n--- 2. Debugging Faulty Snippet (Factorial Fix) ---")
    gen_code_2 = generate_code(tokenizer, model, buggy_code, max_new_tokens=60)
    print(gen_code_2)

    report = [
        "Ex. No: 7 - AI-Powered Code Generation and Debugging Assistant",
        f"Model: {MODEL_NAME}",
        "",
        "=== 1. Code Generation from Natural-Language Prompt ===",
        "PROMPT:",
        prompt1,
        "",
        "GENERATED FUNCTION:",
        gen_code_1,
        "",
        "=== 2. Debugging Faulty Snippet ===",
        "BUGGY CODE:",
        buggy_code,
        "",
        "DEBUG SUGGESTION / REPAIRED CODE:",
        gen_code_2,
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"\nOutput written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
