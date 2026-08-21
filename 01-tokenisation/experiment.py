import json
from pathlib import Path
from datetime import datetime

# pyrefly: ignore [missing-import]
import tiktoken
import pandas as pd
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

INPUT_FILE = Path(__file__).parent / "inputs.json"
OUTPUT_DIR = Path(__file__).parent / "results"
COMPARISONS_DIR = OUTPUT_DIR / "comparisons"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

_last_input_text = None

# ---------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------
def load_tokenizers():

    tokenizers = {}

    # OpenAI
    tokenizers["openai_o200k_base"] = tiktoken.get_encoding("o200k_base")
    tokenizers["openai_o200k_harmony"] = tiktoken.get_encoding("o200k_harmony")
    tokenizers["openai_cl100k"] = tiktoken.get_encoding("cl100k_base")

    # Qwen
    tokenizers["qwen2.5"] = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizers["llama3"] = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizers["smollm2"] = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizers["gemma3"] = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    tokenizers["gpt-oss-20b"] = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

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

def show_tokens(tokenizer_name, tokenizer, text, output_file=None):

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokens_file = OUTPUT_DIR / f"tokens_{timestamp}.txt"
    tokens_file.write_text("", encoding="utf-8")

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

                show_tokens(tokenizer_name, tokenizer, text, output_file=tokens_file)

    df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / f"tokenisation_results_{timestamp}.csv"

    df.to_csv(output_file, index=False)

    print("\nExperiment complete.")
    print(f"Results written to: {output_file}")
    print(f"Tokens written to: {tokens_file}")

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
    qwen_vs_o200k_df = pivot[["qwen2.5", "openai_o200k_base"]].copy()
    qwen_vs_o200k_df["qwen_vs_o200k"] = (
        qwen_vs_o200k_df["qwen2.5"] / qwen_vs_o200k_df["openai_o200k_base"] - 1
    ) * 100
    qwen_vs_o200k_file = COMPARISONS_DIR / f"qwen_vs_o200k_{timestamp}.csv"
    qwen_vs_o200k_df.round(2).to_csv(qwen_vs_o200k_file)
    print(f"Qwen2.5 vs OpenAI O200K comparison written to: {qwen_vs_o200k_file}")

    # 2. cl100k_vs_o200k
    cl100k_vs_o200k_df = pivot[["openai_cl100k", "openai_o200k_base"]].copy()
    cl100k_vs_o200k_df["cl100k_vs_o200k"] = (
        cl100k_vs_o200k_df["openai_cl100k"] / cl100k_vs_o200k_df["openai_o200k_base"] - 1
    ) * 100
    cl100k_vs_o200k_file = COMPARISONS_DIR / f"cl100k_vs_o200k_{timestamp}.csv"
    cl100k_vs_o200k_df.round(2).to_csv(cl100k_vs_o200k_file)
    print(f"OpenAI CL100k vs OpenAI O200K comparison written to: {cl100k_vs_o200k_file}")

    # 3. o200k_base vs o200k_harmony
    o200k_base_vs_o200k_harmony_df = pivot[["openai_o200k_base", "openai_o200k_harmony"]].copy()
    o200k_base_vs_o200k_harmony_df["o200k_base_vs_o200k_harmony"] = (
        o200k_base_vs_o200k_harmony_df["openai_o200k_base"] / o200k_base_vs_o200k_harmony_df["openai_o200k_harmony"] - 1
    ) * 100
    o200k_base_vs_o200k_harmony_file = COMPARISONS_DIR / f"o200k_base_vs_o200k_harmony_{timestamp}.csv"
    o200k_base_vs_o200k_harmony_df.round(2).to_csv(o200k_base_vs_o200k_harmony_file)
    print(f"OpenAI O200k Base vs OpenAI O200k Harmony comparison written to: {o200k_base_vs_o200k_harmony_file}")

    # 4. smollm2_vs_o200k
    smollm2_vs_o200k_df = pivot[["smollm2", "openai_o200k_base"]].copy()
    smollm2_vs_o200k_df["smollm2_vs_o200k"] = (
        smollm2_vs_o200k_df["smollm2"] / smollm2_vs_o200k_df["openai_o200k_base"] - 1
    ) * 100
    smollm2_vs_o200k_file = COMPARISONS_DIR / f"smollm2_vs_o200k_{timestamp}.csv"
    smollm2_vs_o200k_df.round(2).to_csv(smollm2_vs_o200k_file)
    print(f"SmolLM2 vs OpenAI O200K comparison written to: {smollm2_vs_o200k_file}")

    # 5. smollm2_vs_qwen2.5
    smollm2_vs_qwen_df = pivot[["smollm2", "qwen2.5"]].copy()
    smollm2_vs_qwen_df["smollm2_vs_qwen2.5"] = (
        smollm2_vs_qwen_df["smollm2"] / smollm2_vs_qwen_df["qwen2.5"] - 1
    ) * 100
    smollm2_vs_qwen_file = COMPARISONS_DIR / f"smollm2_vs_qwen2.5_{timestamp}.csv"
    smollm2_vs_qwen_df.round(2).to_csv(smollm2_vs_qwen_file)
    print(f"SmolLM2 vs Qwen2.5 comparison written to: {smollm2_vs_qwen_file}")

    #6. o200k_harmony vs gemma3
    gemma3_vs_o200k_df = pivot[["gemma3", "openai_o200k_harmony"]].copy()
    gemma3_vs_o200k_df["gemma3_vs_o200k"] = (
        gemma3_vs_o200k_df["gemma3"] / gemma3_vs_o200k_df["openai_o200k_harmony"] - 1
    ) * 100
    gemma3_vs_o200k_file = COMPARISONS_DIR / f"gemma3_vs_o200k_harmony_{timestamp}.csv"
    gemma3_vs_o200k_df.round(2).to_csv(gemma3_vs_o200k_file)
    print(f"Gemma3 vs OpenAI O200k Harmony comparison written to: {gemma3_vs_o200k_file}")

    #7. gpt-oss-20b vs o200k_harmony
    gpt_oss_20b_vs_o200k_df = pivot[["gpt-oss-20b", "openai_o200k_harmony"]].copy()
    gpt_oss_20b_vs_o200k_df["gpt-oss-20b_vs_o200k_harmony"] = (
        gpt_oss_20b_vs_o200k_df["gpt-oss-20b"] / gpt_oss_20b_vs_o200k_df["openai_o200k_harmony"] - 1
    ) * 100
    gpt_oss_20b_vs_o200k_file = COMPARISONS_DIR / f"gpt-oss-20b_vs_o200k_harmony_{timestamp}.csv"
    gpt_oss_20b_vs_o200k_df.round(2).to_csv(gpt_oss_20b_vs_o200k_file)
    print(f"GPT-oss-20b vs OpenAI O200k Harmony comparison written to: {gpt_oss_20b_vs_o200k_file}")

    #8. gemma3_vs_gpt-oss-20b
    gemma3_vs_gpt_oss_20b_df = pivot[["gemma3", "gpt-oss-20b"]].copy()
    gemma3_vs_gpt_oss_20b_df["gemma3_vs_gpt_oss_20b"] = (
        gemma3_vs_gpt_oss_20b_df["gemma3"] / gemma3_vs_gpt_oss_20b_df["gpt-oss-20b"] - 1
    ) * 100
    gemma3_vs_gpt_oss_20b_file = COMPARISONS_DIR / f"gemma3_vs_gpt_oss_20b_{timestamp}.csv"
    gemma3_vs_gpt_oss_20b_df.round(2).to_csv(gemma3_vs_gpt_oss_20b_file)
    print(f"Gemma3 vs GPT-oss-20b comparison written to: {gemma3_vs_gpt_oss_20b_file}")

if __name__ == "__main__":
    main()