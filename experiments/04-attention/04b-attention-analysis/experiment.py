"""
Experiment 04b — Individual Attention Head Analysis

Purpose:
    Inspect individual attention matrices instead of immediately averaging
    attention across all heads and query positions.

This experiment allows us to investigate:

    model
      ↓
    Transformer layers
      ↓
    individual attention heads
      ↓
    N × N attention matrix
      ↓
    token-to-token relationships

The experiment:
    1. Runs a prompt through a model.
    2. Captures attention weights from every Transformer layer.
    3. Selects individual layers and heads.
    4. Writes the complete attention matrix to a result file.
    5. Identifies the strongest attention targets for each query token.
    6. Saves heatmaps for selected layer/head combinations.
    7. Produces summary statistics for comparing layers and heads.

This deliberately uses eager attention because we need access to the
attention matrices.
"""

from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# Configuration
# ============================================================

MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "gemma3": "google/gemma-3-1b-pt",
    "llama3": "meta-llama/Llama-3.2-1B",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}


PROMPTS = {
    "simple": "The cat sat on the mat.",

    "pronoun": (
        "The cat sat on the mat because it was tired."
    ),

    "long_context": (
        "Sarah gave the book to Emma because she had already "
        "finished reading it."
    ),
}


# ------------------------------------------------------------
# Which layers should we inspect?
#
# Use:
#     "first"
#     "middle"
#     "last"
#
# This gives us a useful early/middle/late comparison without
# producing hundreds of heatmaps.
# ------------------------------------------------------------

LAYERS_TO_ANALYSE = [
    "first",
    "middle",
    "last",
]


# ------------------------------------------------------------
# Which attention heads should we inspect?
#
# "all" will produce matrices/heatmaps for every head in the
# selected layers.
#
# For the first experiment, I recommend a small selection:
#     [0, 1, 2]
#
# Change this to "all" once you are comfortable with the output.
# ------------------------------------------------------------

HEADS_TO_ANALYSE = [0, 1, 2]


# ------------------------------------------------------------
# Number of strongest attention targets to report for each
# query token.
# ------------------------------------------------------------

TOP_K = 3


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================
# Model loading
# ============================================================

def load_model(model_name: str):
    """
    Load tokenizer and model.

    We explicitly use eager attention because the experiment needs
    access to attention weights.
    """

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    )

    model.eval()

    return tokenizer, model


# ============================================================
# Attention extraction
# ============================================================

def analyse_attention(
    prompt: str,
    tokenizer,
    model,
):
    """
    Run a prompt through the model and return:

        tokens
        attention tensors

    Attention tensors have the expected conceptual structure:

        [batch, heads, query_positions, key_positions]
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

    if attentions is None or len(attentions) == 0:
        raise RuntimeError(
            "No attention tensors were returned. "
            "Make sure attn_implementation='eager' "
            "is being used."
        )

    return tokens, attentions


# ============================================================
# Layer selection
# ============================================================

def resolve_layer_indices(
    num_layers: int,
    selections: list[str],
):
    """
    Convert human-readable layer selections into actual indices.

    Example:

        30 layers

        first  -> 0
        middle -> 15
        last   -> 29
    """

    indices = []

    for selection in selections:

        if selection == "first":
            index = 0

        elif selection == "middle":
            index = num_layers // 2

        elif selection == "last":
            index = num_layers - 1

        else:
            raise ValueError(
                f"Unknown layer selection: {selection}"
            )

        if index not in indices:
            indices.append(index)

    return indices


# ============================================================
# Attention matrix extraction
# ============================================================

def get_attention_matrix(
    attentions,
    layer_index: int,
    head_index: int,
):
    """
    Extract one attention matrix.

    Original tensor:

        [batch, heads, query, key]

    We have batch size 1, so:

        attentions[layer][0, head]

    gives:

        [query, key]

    """

    layer_attention = attentions[layer_index]

    matrix = layer_attention[
        0,
        head_index,
    ]

    return matrix.detach().cpu().float().numpy()


# ============================================================
# Top attention targets
# ============================================================

def get_top_attention_targets(
    matrix: np.ndarray,
    tokens: list[str],
    top_k: int,
):
    """
    For each query token, identify the key tokens receiving
    the highest attention.

    Returns:

        [
            {
                "query": ...,
                "targets": [
                    (token, position, weight),
                    ...
                ]
            }
        ]
    """

    results = []

    for query_index, query_token in enumerate(tokens):

        row = matrix[query_index]

        # Sort highest → lowest.
        indices = np.argsort(row)[::-1]

        targets = []

        for key_index in indices[:top_k]:

            targets.append(
                (
                    tokens[key_index],
                    int(key_index),
                    float(row[key_index]),
                )
            )

        results.append(
            {
                "query_index": query_index,
                "query_token": query_token,
                "targets": targets,
            }
        )

    return results


# ============================================================
# Matrix statistics
# ============================================================

def calculate_matrix_statistics(
    matrix: np.ndarray,
):
    """
    Calculate simple statistics for one attention matrix.
    """

    return {
        "mean": float(matrix.mean()),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "std": float(matrix.std()),
    }


# ============================================================
# Heatmap
# ============================================================

def save_attention_heatmap(
    matrix: np.ndarray,
    tokens: list[str],
    model_name: str,
    prompt_name: str,
    layer_index: int,
    head_index: int,
    output_dir: Path,
):
    """
    Save an attention matrix as a heatmap.

    One heatmap = one attention head in one Transformer layer.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{prompt_name}"
        f"_layer-{layer_index + 1}"
        f"_head-{head_index + 1}"
        f".png"
    )

    output_path = output_dir / filename

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix,
        aspect="auto",
    )

    plt.colorbar(
        label="Attention weight"
    )

    plt.xticks(
        range(len(tokens)),
        tokens,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(tokens)),
        tokens,
    )

    plt.xlabel(
        "Key token"
    )

    plt.ylabel(
        "Query token"
    )

    plt.title(
        f"{model_name} | "
        f"{prompt_name} | "
        f"Layer {layer_index + 1} | "
        f"Head {head_index + 1}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# ============================================================
# Write one attention matrix
# ============================================================

def write_attention_matrix(
    file,
    matrix: np.ndarray,
    tokens: list[str],
):
    """
    Write the complete N × N matrix as a Markdown table.
    """

    file.write(
        "### Attention Matrix\n\n"
    )

    file.write(
        "Rows represent **query tokens** and columns represent "
        "**key tokens**.\n\n"
    )

    # Header
    file.write("| Query / Key |")

    for token in tokens:
        file.write(
            f" {token} |"
        )

    file.write("\n")

    file.write("|---|")

    for _ in tokens:
        file.write("---:|")

    file.write("\n")

    # Rows
    for row_index, token in enumerate(tokens):

        file.write(
            f"| {token} |"
        )

        for column_index in range(len(tokens)):

            value = matrix[
                row_index,
                column_index,
            ]

            file.write(
                f" {value:.4f} |"
            )

        file.write("\n")

    file.write("\n")


# ============================================================
# Write top attention targets
# ============================================================

def write_top_attention_targets(
    file,
    results,
):
    """
    Write the strongest attention targets for each query token.
    """

    file.write(
        "### Strongest Attention Targets\n\n"
    )

    file.write(
        "For each query token, the table shows the top attention "
        "targets for this head.\n\n"
    )

    file.write(
        "| Query | Top attention targets |\n"
    )

    file.write(
        "|---|---|\n"
    )

    for result in results:

        query = result["query_token"]

        targets = []

        for (
            token,
            position,
            weight,
        ) in result["targets"]:

            targets.append(
                f"`{token}` "
                f"(position {position}, "
                f"{weight:.4f})"
            )

        file.write(
            f"| `{query}` | "
            f"{', '.join(targets)} |\n"
        )

    file.write("\n")


# ============================================================
# Run one prompt
# ============================================================

def analyse_prompt(
    prompt_name: str,
    prompt: str,
    tokens,
    attentions,
    model_name: str,
    output_file,
    heatmap_dir: Path,
):
    """
    Analyse selected layers and heads for one prompt.
    """

    num_layers = len(attentions)

    num_heads = attentions[
        0
    ].shape[1]

    sequence_length = len(tokens)

    layer_indices = resolve_layer_indices(
        num_layers,
        LAYERS_TO_ANALYSE,
    )

    # --------------------------------------------------------
    # Prompt metadata
    # --------------------------------------------------------

    output_file.write(
        f"## Experiment: {prompt_name}\n\n"
    )

    output_file.write(
        "### Input\n\n"
    )

    output_file.write(
        "```text\n"
    )

    output_file.write(
        prompt
    )

    output_file.write(
        "\n```\n\n"
    )

    output_file.write(
        "### Tokens\n\n"
    )

    for index, token in enumerate(tokens):

        output_file.write(
            f"{index}: `{token}`  \n"
        )

    output_file.write("\n")

    output_file.write(
        "### Model / Attention Information\n\n"
    )

    output_file.write(
        f"- Transformer layers: {num_layers}\n"
    )

    output_file.write(
        f"- Attention heads: {num_heads}\n"
    )

    output_file.write(
        f"- Sequence length: {sequence_length}\n"
    )

    output_file.write(
        f"- Attention tensor shape: "
        f"`[1, {num_heads}, "
        f"{sequence_length}, "
        f"{sequence_length}]`\n\n"
    )

    # --------------------------------------------------------
    # Selected layers
    # --------------------------------------------------------

    output_file.write(
        "### Selected Layers\n\n"
    )

    output_file.write(
        "The experiment analyses:\n\n"
    )

    for layer_index in layer_indices:

        output_file.write(
            f"- Layer {layer_index + 1}\n"
        )

    output_file.write("\n")

    # --------------------------------------------------------
    # Analyse layers
    # --------------------------------------------------------

    for layer_index in layer_indices:

        output_file.write(
            f"---\n\n"
        )

        output_file.write(
            f"## Layer {layer_index + 1}\n\n"
        )

        available_heads = attentions[
            layer_index
        ].shape[1]

        # ----------------------------------------------------
        # Resolve heads
        # ----------------------------------------------------

        if HEADS_TO_ANALYSE == "all":

            head_indices = list(
                range(available_heads)
            )

        else:

            head_indices = [
                head
                for head in HEADS_TO_ANALYSE
                if head < available_heads
            ]

        # ----------------------------------------------------
        # Individual heads
        # ----------------------------------------------------

        for head_index in head_indices:

            matrix = get_attention_matrix(
                attentions,
                layer_index,
                head_index,
            )

            statistics = calculate_matrix_statistics(
                matrix
            )

            top_targets = get_top_attention_targets(
                matrix,
                tokens,
                TOP_K,
            )

            output_file.write(
                f"### Head {head_index + 1}\n\n"
            )

            output_file.write(
                f"Matrix shape: "
                f"`{matrix.shape[0]} × "
                f"{matrix.shape[1]}`\n\n"
            )

            output_file.write(
                "#### Matrix Statistics\n\n"
            )

            output_file.write(
                f"- Mean: "
                f"{statistics['mean']:.6f}\n"
            )

            output_file.write(
                f"- Minimum: "
                f"{statistics['min']:.6f}\n"
            )

            output_file.write(
                f"- Maximum: "
                f"{statistics['max']:.6f}\n"
            )

            output_file.write(
                f"- Standard deviation: "
                f"{statistics['std']:.6f}\n\n"
            )

            write_attention_matrix(
                output_file,
                matrix,
                tokens,
            )

            write_top_attention_targets(
                output_file,
                top_targets,
            )

            # ------------------------------------------------
            # Heatmap
            # ------------------------------------------------

            save_attention_heatmap(
                matrix=matrix,
                tokens=tokens,
                model_name=model_name,
                prompt_name=prompt_name,
                layer_index=layer_index,
                head_index=head_index,
                output_dir=heatmap_dir,
            )


# ============================================================
# Run one complete model
# ============================================================

def run_experiment(
    model_key: str,
    model_name: str,
):
    """
    Run the complete 04b experiment for one model.
    """

    print()
    print("=" * 70)
    print(f"MODEL: {model_key}")
    print(model_name)
    print("=" * 70)

    tokenizer, model = load_model(
        model_name
    )

    model_results_dir = (
        RESULTS_DIR / model_key
    )

    heatmap_dir = (
        model_results_dir / "heatmaps"
    )

    model_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file = (
        model_results_dir /
        "attention_analysis.md"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Attention Matrix Analysis\n\n"
        )

        file.write(
            f"**Model:** `{model_name}`\n\n"
        )

        file.write(
            "This experiment inspects individual "
            "attention heads rather than averaging "
            "the attention tensor immediately.\n\n"
        )

        for prompt_name, prompt in PROMPTS.items():

            print(
                f"Analysing prompt: {prompt_name}"
            )

            tokens, attentions = analyse_attention(
                prompt,
                tokenizer,
                model,
            )

            analyse_prompt(
                prompt_name=prompt_name,
                prompt=prompt,
                tokens=tokens,
                attentions=attentions,
                model_name=model_key,
                output_file=file,
                heatmap_dir=heatmap_dir,
            )

    print()
    print(
        f"Results written to:\n{result_file}"
    )

    print(
        f"Heatmaps written to:\n{heatmap_dir}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Start with ONE model.
    #
    # Change this to another model once you understand the
    # generated results.
    # --------------------------------------------------------

    MODEL_KEY = "smollm2"

    run_experiment(
        model_key=MODEL_KEY,
        model_name=MODELS[MODEL_KEY],
    )