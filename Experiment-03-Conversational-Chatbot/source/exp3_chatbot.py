"""
Ex. No: 3
CONVERSATIONAL AI CHATBOT USING TRANSFORMER-BASED LANGUAGE MODELS

Non-interactive by default: reads turns from input/conversation.txt.
Pass --interactive to chat live in the terminal.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "input" / "conversation.txt"
OUTPUT_FILE = ROOT / "output" / "chat_transcript.txt"
MODEL_NAME = "microsoft/DialoGPT-small"


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def bot_reply(tokenizer, model, chat_history_ids, user_input):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )
    chat_history_ids = model.generate(
        bot_input_ids,
        max_new_tokens=80,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
    )
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1] :][0],
        skip_special_tokens=True,
    )
    return chat_history_ids, response


def run_scripted(tokenizer, model) -> None:
    turns = [
        line.strip()
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip().lower() != "quit"
    ]
    transcript = [
        "Ex. No: 3 - Conversational AI Chatbot (DialoGPT)",
        f"Model: {MODEL_NAME}",
        "Chatbot ready! Type 'quit' to exit.",
        "",
    ]
    chat_history_ids = None
    print("Chatbot ready! Type 'quit' to exit.")
    for user_input in turns:
        print(f">> User: {user_input}")
        transcript.append(f">> User: {user_input}")
        chat_history_ids, response = bot_reply(tokenizer, model, chat_history_ids, user_input)
        print(f"Bot: {response}")
        transcript.append(f"Bot: {response}")
        transcript.append("")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(transcript), encoding="utf-8")
    print(f"Transcript written to {OUTPUT_FILE}")


def run_interactive(tokenizer, model) -> None:
    print("Chatbot ready! Type 'quit' to exit.")
    chat_history_ids = None
    for _ in range(5):
        user_input = input(">> User: ")
        if user_input.lower() == "quit":
            break
        chat_history_ids, response = bot_reply(tokenizer, model, chat_history_ids, user_input)
        print(f"Bot: {response}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    tokenizer, model = load_model()
    if args.interactive:
        run_interactive(tokenizer, model)
    else:
        run_scripted(tokenizer, model)


if __name__ == "__main__":
    main()
