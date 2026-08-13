import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import tiktoken
import pandas as pd
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

INPUT_FILE = Path(__file__).parent / "inputs.json"
OUTPUT_DIR = Path(__file__).parents[2] / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
TOKENS_FILE = OUTPUT_DIR / "tokens.txt"

_last_input_text = None

# ---------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------
def load_tokenizers():

    tokenizers = {}

    # OpenAI
    tokenizers["openai_o200k"] = tiktoken.get_encoding("o200k_base")
    tokenizers["openai_cl100k"] = tiktoken.get_encoding("cl100k_base")
    

    # Qwen
    tokenizers["qwen2.5"] = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    #tokenizers["llama3"] = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizers["smollm2"] = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    return tokenizers


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def count_words(text):
    return len(text.split())

def count_characters(text):
    return len(text)

def tokenise(tokenizer, text):
    if hasattr(tokenizer, "encode"):

        # tiktoken
        if tokenizer.__class__.__module__.startswith("tiktoken"):
            return tokenizer.encode(text)

        # Hugging Face tokenizer
        return tokenizer.encode(text, add_special_tokens=False)

    raise ValueError("Unsupported tokenizer")

def calculate_metrics(tokenizer_name, tokenizer, category, text):

    tokens = tokenise(tokenizer, text)

    characters = count_characters(text)
    words = count_words(text)
    token_count = len(tokens)

    return {
        "tokenizer": tokenizer_name,
        "input_type": category,
        "characters": characters,
        "words": words,
        "tokens": token_count,
        "characters_per_token": (
            characters / token_count if token_count else 0
        ),
        "words_per_token": (
            words / token_count if token_count else 0
        ),
        "tokens_per_word": (
            token_count / words if words else 0
        ),
    }

def show_tokens(tokenizer_name, tokenizer, text, output_file=TOKENS_FILE):

    global _last_input_text

    tokens = tokenise(tokenizer, text)

    if tokenizer_name.startswith("openai"):
        pieces = [
            tokenizer.decode([token])
            for token in tokens
        ]
    else:
        pieces = tokenizer.convert_ids_to_tokens(tokens)

    with open(output_file, "a", encoding="utf-8") as f:
        if text != _last_input_text:
            if _last_input_text is not None:
                f.write("\n")
            f.write(f"Input = {text}\n")
            _last_input_text = text

        f.write(f"{tokenizer_name} = {pieces}\n")

# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    global _last_input_text
    _last_input_text = None
    TOKENS_FILE.write_text("", encoding="utf-8")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizers = load_tokenizers()

    results = []

    for category, items in data.items():

        for item_index, text in enumerate(items, start=1):

            for tokenizer_name, tokenizer in tokenizers.items():

                result = calculate_metrics(
                    tokenizer_name,
                    tokenizer,
                    category,
                    text,
                )

                result["item_index"] = item_index

                results.append(result)

                print("\n\nResult")
                print(result)

                show_tokens(tokenizer_name, tokenizer, text)

    df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "tokenisation_results.csv"

    df.to_csv(output_file, index=False)

    print("\nExperiment complete.")
    print(f"Results written to: {output_file}")
    print(f"Tokens written to: {TOKENS_FILE}")

    print("\nSummary:")
    print(
        df.groupby(["input_type", "tokenizer"])["tokens"]
        .mean()
        .round(2)
        .to_string()
    )
    print("\n")

    pivot = df.pivot_table(
        index=["input_type", "item_index"],
        columns="tokenizer",
        values="tokens"
    )

    # 1. qwen_vs_o200k
    qwen_vs_o200k_df = pivot[["qwen2.5", "openai_o200k"]].copy()
    qwen_vs_o200k_df["qwen_vs_o200k"] = (
        qwen_vs_o200k_df["qwen2.5"] / qwen_vs_o200k_df["openai_o200k"] - 1
    ) * 100
    qwen_vs_o200k_file = OUTPUT_DIR / "qwen_vs_o200k.csv"
    qwen_vs_o200k_df.round(2).to_csv(qwen_vs_o200k_file)
    print(f"Qwen2.5 vs OpenAI O200K comparison written to: {qwen_vs_o200k_file}")

    # 2. cl100k_vs_o200k
    cl100k_vs_o200k_df = pivot[["openai_cl100k", "openai_o200k"]].copy()
    cl100k_vs_o200k_df["cl100k_vs_o200k"] = (
        cl100k_vs_o200k_df["openai_cl100k"] / cl100k_vs_o200k_df["openai_o200k"] - 1
    ) * 100
    cl100k_vs_o200k_file = OUTPUT_DIR / "cl100k_vs_o200k.csv"
    cl100k_vs_o200k_df.round(2).to_csv(cl100k_vs_o200k_file)
    print(f"OpenAI CL100k vs OpenAI O200K comparison written to: {cl100k_vs_o200k_file}")

    # 3. smollm2_vs_o200k
    smollm2_vs_o200k_df = pivot[["smollm2", "openai_o200k"]].copy()
    smollm2_vs_o200k_df["smollm2_vs_o200k"] = (
        smollm2_vs_o200k_df["smollm2"] / smollm2_vs_o200k_df["openai_o200k"] - 1
    ) * 100
    smollm2_vs_o200k_file = OUTPUT_DIR / "smollm2_vs_o200k.csv"
    smollm2_vs_o200k_df.round(2).to_csv(smollm2_vs_o200k_file)
    print(f"SmolLM2 vs OpenAI O200K comparison written to: {smollm2_vs_o200k_file}")

    # 4. smollm2_vs_qwen2.5
    smollm2_vs_qwen_df = pivot[["smollm2", "qwen2.5"]].copy()
    smollm2_vs_qwen_df["smollm2_vs_qwen2.5"] = (
        smollm2_vs_qwen_df["smollm2"] / smollm2_vs_qwen_df["qwen2.5"] - 1
    ) * 100
    smollm2_vs_qwen_file = OUTPUT_DIR / "smollm2_vs_qwen2.5.csv"
    smollm2_vs_qwen_df.round(2).to_csv(smollm2_vs_qwen_file)
    print(f"SmolLM2 vs Qwen2.5 comparison written to: {smollm2_vs_qwen_file}")


if __name__ == "__main__":
    main()