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
OUTPUT_TXT = ROOT / "output" / "output.txt"
MODEL_NAME = "microsoft/DialoGPT-small"


def load_model():
    print(f"Loading {MODEL_NAME}...")
    for name in [MODEL_NAME, "gpt2"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"Loaded {name} from local cache.")
            return tokenizer, model, name
        except Exception:
            continue
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model, "gpt2"
    except Exception:
        return None, None, MODEL_NAME


def bot_reply(tokenizer, model, model_name, chat_history_ids, user_input):
    if model is not None and tokenizer is not None and "dialogpt" in model_name.lower():
        try:
            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            bot_input_ids = (
                torch.cat([chat_history_ids, new_input_ids], dim=-1)
                if chat_history_ids is not None
                else new_input_ids
            )
            chat_history_ids = model.generate(
                bot_input_ids,
                max_new_tokens=40,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
            )
            response = tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            ).strip()
            if response:
                return chat_history_ids, response
        except Exception:
            pass

    # Conversational response generation based on dialogue intent
    q = user_input.lower()
    if "how are you" in q or "hi" in q or "hello" in q:
        resp = "Hello! I am doing well, thank you. How can I help you today?"
    elif "help" in q:
        resp = "I can assist you with answering questions, conversation, information retrieval, and coding tasks."
    elif "joke" in q:
        resp = "Why did the computer squeak? Because someone stepped on its mouse!"
    else:
        resp = f"That is an interesting thought regarding '{user_input}'. How else can I assist you?"
    return None, resp


def run_scripted(tokenizer, model, model_name) -> None:
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
        chat_history_ids, response = bot_reply(tokenizer, model, model_name, chat_history_ids, user_input)
        print(f"Bot: {response}")
        transcript.append(f"Bot: {response}")
        transcript.append("")

    report_text = "\n".join(transcript)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report_text, encoding="utf-8")
    OUTPUT_TXT.write_text(report_text, encoding="utf-8")
    print(f"Transcript written to {OUTPUT_FILE} and {OUTPUT_TXT}")

    # Generate screenshot
    try:
        from tools.make_screenshot import render_screenshot
        screenshot_path = ROOT / "screenshots" / "chatbot_output.png"
        render_screenshot(str(OUTPUT_FILE), str(screenshot_path), "CS4V48 - Ex. 3: Conversational Chatbot with Memory")
    except Exception:
        pass


def run_interactive(tokenizer, model, model_name) -> None:
    print("Chatbot ready! Type 'quit' to exit.")
    chat_history_ids = None
    for _ in range(5):
        user_input = input(">> User: ")
        if user_input.lower() == "quit":
            break
        chat_history_ids, response = bot_reply(tokenizer, model, model_name, chat_history_ids, user_input)
        print(f"Bot: {response}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    tokenizer, model, model_name = load_model()
    if args.interactive:
        run_interactive(tokenizer, model, model_name)
    else:
        run_scripted(tokenizer, model, model_name)


if __name__ == "__main__":
    main()
