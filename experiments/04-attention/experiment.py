"""
Attention Experiment
====================

Goal:
    Inspect how a Transformer distributes attention between tokens.

This experiment uses Hugging Face Transformers directly rather than
Ollama because we want access to the model's internal attention
weights.

The experiment:
    1. Tokenises a sentence.
    2. Runs the sentence through the Transformer.
    3. Extracts attention weights.
    4. Saves the attention information to a Markdown result file.

Default model:
    HuggingFaceTB/SmolLM2-135M

Runtime:
    Hugging Face Transformers + PyTorch
"""

from datetime import datetime
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#MODEL = "HuggingFaceTB/SmolLM2-135M"
MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "gemma3": "google/gemma-3-1b-pt",
    "llama3": "meta-llama/Llama-3.2-1B",
    "gpt-oss-20b": "openai/gpt-oss-20b"
}

RESULTS_DIR = Path(__file__).parent / "results"

PROMPTS = {
    "simple": "The cat sat on the mat.",

    "pronoun": (
        "The cat sat on the mat because it was tired."
    ),

    "longer_context": (
        "Sarah gave the book to Emma because she had already "
        "finished reading it."
    ),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load tokenizer and model with attention output enabled."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager"
        # We need the actual attention matrices for this experiment.
        # SDPA does not return them when output_attentions=True,
        # so we explicitly use the eager implementation.
    )

    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def analyse_attention(
    prompt: str,
    tokenizer,
    model,
):
    """
    Run the prompt through the model and return token and
    attention information.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
        )

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    attentions = outputs.attentions

    return tokens, attentions


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_attention_summary(
    tokens,
    attentions,
):
    """
    Create a compact textual summary of attention.

    We inspect the final Transformer layer and calculate the
    average attention received by each token.
    """
    
    final_layer = attentions[-1]

    # Shape:
    # [batch, heads, sequence, sequence]

    attention = final_layer[0]

    # Average across attention heads.
    average_attention = attention.mean(dim=0)

    # Average attention received by each token.
    attention_received = average_attention.mean(dim=0)

    ranking = sorted(
        zip(tokens, attention_received.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranking


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model_name) -> Path:
    """Run the attention experiment and save results."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_filename = model_name.replace("/", "_")

    result_file = (
        RESULTS_DIR
        / f"results_{model_filename}_{timestamp}.md"
    )

    tokenizer, model = load_model(model_name)

    with result_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("# Attention Experiment Results\n\n")

        file.write(
            f"**Date:** {datetime.now().isoformat()}\n\n"
        )

        file.write(
            f"**Model:** `{model_name}`\n\n"
        )

        file.write("---\n\n")

        for name, prompt in PROMPTS.items():

            file.write(
                f"## Experiment: {name}\n\n"
            )

            file.write("### Input\n\n")
            file.write("```text\n")
            file.write(prompt)
            file.write("\n```\n\n")

            tokens, attentions = analyse_attention(
                prompt,
                tokenizer,
                model,
            )

            print("Type of attentions", type(attentions))
            print("Number of attention outputs:", len(attentions))
            print("Type of last layer", type(attentions[-1]))
            print("Shape of last layer", attentions[-1].shape)

            for layer_index, attention in enumerate(attentions):
                print(
                    f"Layer {layer_index}: "
                    f"type={type(attention)}, "
                    f"shape={getattr(attention, 'shape', None)}"
                )

            file.write("### Tokens\n\n")

            for index, token in enumerate(tokens):
                file.write(
                    f"{index}: `{token}`  \n"
                )

            file.write("\n")

            file.write("### Attention information\n\n")

            file.write(
                f"- Number of Transformer layers: "
                f"{len(attentions)}\n"
            )

            file.write(
                f"- Number of attention heads in final layer: "
                f"{attentions[-1].shape[1]}\n"
            )

            file.write(
                f"- Sequence length: "
                f"{attentions[-1].shape[-1]}\n\n"
            )

            file.write(
                "### Average attention received by token\n\n"
            )

            file.write(
                "| Token | Average attention |\n"
            )
            file.write(
                "|---|---:|\n"
            )

            ranking = format_attention_summary(
                tokens,
                attentions,
            )

            for token, score in ranking:
                file.write(
                    f"| `{token}` | {score:.6f} |\n"
                )

            file.write("\n---\n\n")

    return result_file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    for model_name, model_id in MODELS.items():
        result_file = run_experiment(model_id)
        print(f"Results saved to: {result_file}")