"""
Experiment 01 — Inference Basics: Cross-Model Performance

Purpose
-------
Compare inference behaviour of several local Hugging Face causal language
models under the same workload and runtime conditions.

The experiment focuses on architecturally useful inference metrics:

    - input token count
    - output token count
    - total generation latency
    - output tokens / second
    - repeated-run variability

Workloads
---------
A. short input  -> short output
B. short input  -> long output
C. long input   -> short output
D. long input   -> long output

Models
------
By default:

    Qwen/Qwen2.5-0.5B-Instruct
    google/gemma-3-1b-it
    meta-llama/Llama-3.2-1B-Instruct
    openai/gpt-oss-20b

IMPORTANT
---------
Cross-model comparisons are meaningful only if the models are run under
comparable hardware/runtime/precision conditions.

The script records the runtime environment so that results can be interpreted
correctly.

Some models may require Hugging Face authentication or substantial memory.

TTFT
----
This experiment measures TOTAL generation latency and output throughput.

It does NOT claim to measure true production TTFT (Time To First Token).

True TTFT requires streaming/token-level measurement and will be covered
separately.

QUALITY
-------
Quality is deliberately NOT auto-scored.

The CSV contains:

    quality_score_manual

You can manually score each response from 1–5 and rerun the chart generation
if you want to create a quality-vs-latency comparison.

Usage
-----
    python experiment.py

Optional environment variables:

    MODEL_NAMES="model-a,model-b,model-c"
    RUNS=3
    WARMUP_RUNS=1
    MAX_NEW_TOKENS_SHORT=32
    MAX_NEW_TOKENS_LONG=128
    DEVICE=auto

Example:

    MODEL_NAMES="Qwen/Qwen2.5-0.5B-Instruct,google/gemma-3-1b-it" \
    python experiment.py
"""

import csv
import gc
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "openai/gpt-oss-20b",
]


MODEL_NAMES = [
    model.strip()
    for model in os.getenv(
        "MODEL_NAMES",
        ",".join(DEFAULT_MODELS),
    ).split(",")
    if model.strip()
]


RUNS = int(os.getenv("RUNS", "3"))

WARMUP_RUNS = int(
    os.getenv("WARMUP_RUNS", "1")
)

MAX_NEW_TOKENS_SHORT = int(
    os.getenv("MAX_NEW_TOKENS_SHORT", "32")
)

MAX_NEW_TOKENS_LONG = int(
    os.getenv("MAX_NEW_TOKENS_LONG", "128")
)


# ============================================================================
# WORKLOADS
# ============================================================================

WORKLOADS = [

    {
        "workload_id": "A",
        "name": "short_input_short_output",

        "prompt": (
            "In one sentence, explain what an API gateway does."
        ),

        "max_new_tokens": MAX_NEW_TOKENS_SHORT,
    },

    {
        "workload_id": "B",
        "name": "short_input_long_output",

        "prompt": (
            "Explain the role of an API gateway in an enterprise AI "
            "architecture. Cover routing, authentication, rate limiting, "
            "observability, cost controls, retries, and model selection."
        ),

        "max_new_tokens": MAX_NEW_TOKENS_LONG,
    },

    {
        "workload_id": "C",
        "name": "long_input_short_output",

        "prompt": (
            "Summarise this in one sentence: An enterprise application "
            "sends requests to an AI gateway. The gateway authenticates "
            "callers, applies quotas, selects an appropriate model, "
            "retrieves context, invokes tools where permitted, records "
            "telemetry, applies safety policies, and returns the model "
            "response. The gateway may also enforce tenant isolation, "
            "model access controls, fallback rules, caching, tracing, "
            "and usage budgets."
        ),

        "max_new_tokens": MAX_NEW_TOKENS_SHORT,
    },

    {
        "workload_id": "D",
        "name": "long_input_long_output",

        "prompt": (
            "Explain this architecture and its trade-offs in detail: "
            "An enterprise application sends requests to an AI gateway. "
            "The gateway authenticates callers, applies quotas, selects "
            "an appropriate model, retrieves context, invokes tools where "
            "permitted, records telemetry, applies safety policies, and "
            "returns the model response. The system supports models with "
            "different latency, cost and capability characteristics. "
            "It may use RAG, model routing, caching and fallback models. "
            "Discuss quality, latency, throughput, cost, reliability "
            "and operational complexity."
        ),

        "max_new_tokens": MAX_NEW_TOKENS_LONG,
    },
]


# ============================================================================
# DEVICE
# ============================================================================

def get_device():

    override = os.getenv(
        "DEVICE",
        "auto",
    ).lower()

    if override != "auto":
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def device_name():

    if DEVICE.type == "cuda":

        return torch.cuda.get_device_name(
            DEVICE.index or 0
        )

    if DEVICE.type == "mps":

        return "Apple Metal (MPS)"

    return "CPU"


def synchronise():

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name):

    print(
        f"\nLoading model: {model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if DEVICE.type == "cuda":

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )

        model.to(DEVICE)

    model.eval()

    return tokenizer, model


# ============================================================================
# PROMPT HANDLING
# ============================================================================

def format_prompt(
    tokenizer,
    prompt,
):

    if getattr(
        tokenizer,
        "chat_template",
        None,
    ):

        return tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def get_model_input_device(model):

    """
    For models loaded using device_map="auto", the model may be distributed
    across multiple devices.

    The first non-meta parameter is used as the input device.
    """

    for parameter in model.parameters():

        if parameter.device.type != "meta":

            return parameter.device

    return DEVICE


# ============================================================================
# INFERENCE
# ============================================================================

def generate_once(
    tokenizer,
    model,
    prompt,
    max_new_tokens,
):

    formatted_prompt = format_prompt(
        tokenizer,
        prompt,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    input_device = get_model_input_device(
        model
    )

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    input_tokens = inputs[
        "input_ids"
    ].shape[1]

    synchronise()

    start_time = time.perf_counter()

    with torch.inference_mode():

        output = model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=False,

            use_cache=True,

            pad_token_id=tokenizer.eos_token_id,
        )

    synchronise()

    end_time = time.perf_counter()

    latency_seconds = (
        end_time - start_time
    )

    output_tokens = max(
        output.shape[1] - input_tokens,
        0,
    )

    output_tokens_per_second = (

        output_tokens / latency_seconds

        if latency_seconds > 0

        else 0
    )

    generated_text = tokenizer.decode(

        output[0][input_tokens:],

        skip_special_tokens=True,
    )

    return {

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "latency_seconds": latency_seconds,

        "output_tokens_per_second":
            output_tokens_per_second,

        "output": generated_text,
    }


# ============================================================================
# WARMUP
# ============================================================================

def warmup(
    tokenizer,
    model,
):

    for _ in range(WARMUP_RUNS):

        generate_once(

            tokenizer,

            model,

            "Reply with one short sentence: "
            "What is inference?",

            8,
        )


# ============================================================================
# ENVIRONMENT INFORMATION
# ============================================================================

def environment_metadata():

    cuda_device = ""

    if torch.cuda.is_available():

        cuda_device = torch.cuda.get_device_name(
            DEVICE.index or 0
        )

    return {

        "python_version":
            platform.python_version(),

        "pytorch_version":
            torch.__version__,

        "transformers_version":
            transformers.__version__,

        "platform":
            platform.platform(),

        "device":
            str(DEVICE),

        "device_description":
            device_name(),

        "cuda_available":
            torch.cuda.is_available(),

        "cuda_device":
            cuda_device,
    }


# ============================================================================
# SAVE RAW RESULTS
# ============================================================================

def save_results(
    timestamp,
    rows,
    metadata,
    failures,
):

    csv_path = (
        RESULTS_DIR
        / f"results_{timestamp}.csv"
    )

    txt_path = (
        RESULTS_DIR
        / f"results_{timestamp}.txt"
    )


    # ------------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------------

    fieldnames = list(
        rows[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


    # ------------------------------------------------------------------------
    # TXT
    # ------------------------------------------------------------------------

    with txt_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EXPERIMENT 01 — INFERENCE BASICS\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            "Environment\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        for key, value in metadata.items():

            file.write(
                f"{key}: {value}\n"
            )


        file.write("\nConfiguration\n")

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            f"Runs per workload: {RUNS}\n"
        )

        file.write(
            f"Warmup runs: {WARMUP_RUNS}\n"
        )

        file.write(
            "Models:\n"
        )

        for model in MODEL_NAMES:

            file.write(
                f"  - {model}\n"
            )


        # --------------------------------------------------------------------
        # Failures
        # --------------------------------------------------------------------

        if failures:

            file.write(
                "\nModel load failures\n"
            )

            file.write(
                "-" * 80 + "\n"
            )

            for model, error in failures:

                file.write(
                    f"{model}: {error}\n"
                )


        # --------------------------------------------------------------------
        # Raw results
        # --------------------------------------------------------------------

        file.write(
            "\nRaw results\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        for row in rows:

            file.write(
                f"{row}\n"
            )


        # --------------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------------

        file.write(
            "\nSummary by model and workload\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        groups = {}

        for row in rows:

            if row["error"]:

                continue

            key = (
                row["model"],
                row["workload_name"],
            )

            groups.setdefault(
                key,
                [],
            ).append(row)


        for (
            model,
            workload,
        ), group in groups.items():

            latency_values = [

                float(
                    row["latency_seconds"]
                )

                for row in group
            ]

            throughput_values = [

                float(
                    row[
                        "output_tokens_per_second"
                    ]
                )

                for row in group
            ]


            file.write(
                f"\nModel: {model}\n"
            )

            file.write(
                f"Workload: {workload}\n"
            )

            file.write(
                f"Average latency: "
                f"{statistics.mean(latency_values):.4f}s\n"
            )

            file.write(
                f"Median latency: "
                f"{statistics.median(latency_values):.4f}s\n"
            )

            file.write(
                f"Average output tokens/sec: "
                f"{statistics.mean(throughput_values):.2f}\n"
            )


    return csv_path, txt_path


# ============================================================================
# AGGREGATION
# ============================================================================

def build_summary(rows):

    groups = {}

    for row in rows:

        if row["error"]:

            continue

        key = (
            row["model"],
            row["workload_name"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)


    summary = {}

    for key, group in groups.items():

        summary[key] = {

            "input_tokens":
                statistics.mean(
                    int(
                        row["input_tokens"]
                    )
                    for row in group
                ),

            "output_tokens":
                statistics.mean(
                    int(
                        row["output_tokens"]
                    )
                    for row in group
                ),

            "latency":
                statistics.mean(
                    float(
                        row[
                            "latency_seconds"
                        ]
                    )
                    for row in group
                ),

            "tokens_per_second":
                statistics.mean(
                    float(
                        row[
                            "output_tokens_per_second"
                        ]
                    )
                    for row in group
                ),
        }

    return summary


# ============================================================================
# CHART UTILITIES
# ============================================================================

def save_chart(
    figure,
    timestamp,
    filename,
):

    path = (
        CHARTS_DIR
        / f"{filename}_{timestamp}.png"
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


# ============================================================================
# CHART 1
# Latency vs Output Tokens
# ============================================================================

def chart_latency_vs_output_tokens(
    timestamp,
    summary,
    models,
    workloads,
):

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for model in models:

        points = []

        for workload in workloads:

            key = (
                model,
                workload,
            )

            if key in summary:

                points.append(
                    (
                        summary[key][
                            "output_tokens"
                        ],

                        summary[key][
                            "latency"
                        ],
                    )
                )


        if points:

            x_values, y_values = zip(
                *points
            )

            axis.plot(
                x_values,
                y_values,
                marker="o",
                label=model,
            )


    axis.set_title(
        "Inference Latency vs Output Tokens"
    )

    axis.set_xlabel(
        "Average Output Tokens"
    )

    axis.set_ylabel(
        "Average Latency (seconds)"
    )

    axis.legend(
        fontsize=8
    )

    axis.grid(
        True,
        alpha=0.25,
    )


    return save_chart(
        figure,
        timestamp,
        "01_latency_vs_output_tokens",
    )


# ============================================================================
# CHART 2
# Tokens / Second by Model
# ============================================================================

def chart_tokens_per_second(
    timestamp,
    summary,
    models,
    workloads,
):

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    labels = []

    values = []


    for model in models:

        throughput_values = [

            summary[
                (model, workload)
            ]["tokens_per_second"]

            for workload in workloads

            if (
                model,
                workload
            ) in summary
        ]


        if throughput_values:

            labels.append(model)

            values.append(
                statistics.mean(
                    throughput_values
                )
            )


    axis.bar(
        labels,
        values,
    )

    axis.set_title(
        "Average Output Throughput by Model"
    )

    axis.set_xlabel(
        "Model"
    )

    axis.set_ylabel(
        "Output Tokens / Second"
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )


    return save_chart(
        figure,
        timestamp,
        "02_tokens_per_second_by_model",
    )


# ============================================================================
# CHART 3
# Latency vs Input Tokens
# ============================================================================

def chart_latency_vs_input_tokens(
    timestamp,
    summary,
    models,
    workloads,
):

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )


    for model in models:

        points = []

        for workload in workloads:

            key = (
                model,
                workload,
            )

            if key in summary:

                points.append(
                    (
                        summary[key][
                            "input_tokens"
                        ],

                        summary[key][
                            "latency"
                        ],
                    )
                )


        if points:

            x_values, y_values = zip(
                *points
            )

            axis.plot(
                x_values,
                y_values,
                marker="o",
                label=model,
            )


    axis.set_title(
        "Inference Latency vs Input Context Size"
    )

    axis.set_xlabel(
        "Average Input Tokens"
    )

    axis.set_ylabel(
        "Average Latency (seconds)"
    )

    axis.legend(
        fontsize=8
    )

    axis.grid(
        True,
        alpha=0.25,
    )


    return save_chart(
        figure,
        timestamp,
        "03_latency_vs_input_tokens",
    )


# ============================================================================
# CHART 4
# Latency by Workload
# ============================================================================

def chart_latency_by_workload(
    timestamp,
    summary,
    models,
    workloads,
):

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )


    x_positions = list(
        range(len(workloads))
    )

    number_of_models = len(
        models
    )

    bar_width = (
        0.8 / number_of_models
        if number_of_models
        else 0.8
    )


    for index, model in enumerate(
        models
    ):

        values = []

        for workload in workloads:

            key = (
                model,
                workload,
            )

            if key in summary:

                values.append(
                    summary[key][
                        "latency"
                    ]
                )

            else:

                values.append(0)


        offsets = [

            position
            + (
                index
                - (
                    number_of_models - 1
                ) / 2
            )
            * bar_width

            for position
            in x_positions
        ]


        axis.bar(
            offsets,
            values,
            width=bar_width,
            label=model,
        )


    axis.set_title(
        "Latency by Workload"
    )

    axis.set_xlabel(
        "Workload"
    )

    axis.set_ylabel(
        "Average Latency (seconds)"
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        ["A", "B", "C", "D"]
    )

    axis.legend(
        fontsize=8
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.25,
    )


    return save_chart(
        figure,
        timestamp,
        "04_latency_by_workload",
    )


# ============================================================================
# CHART 5
# Quality vs Latency
# ============================================================================

def chart_quality_vs_latency(
    timestamp,
    rows,
):

    quality_rows = []


    for row in rows:

        score = row.get(
            "quality_score_manual",
            "",
        )

        if score in (
            "",
            None,
        ):

            continue


        try:

            quality_rows.append(
                (
                    row["model"],

                    float(
                        row[
                            "latency_seconds"
                        ]
                    ),

                    float(score),
                )
            )

        except (
            ValueError,
            TypeError,
        ):

            continue


    if not quality_rows:

        return None


    grouped = {}


    for (
        model,
        latency,
        quality,
    ) in quality_rows:

        grouped.setdefault(
            model,
            [],
        ).append(
            (
                latency,
                quality,
            )
        )


    figure, axis = plt.subplots(
        figsize=(10, 6)
    )


    for model, points in grouped.items():

        average_latency = statistics.mean(
            point[0]
            for point in points
        )

        average_quality = statistics.mean(
            point[1]
            for point in points
        )


        axis.scatter(
            average_latency,
            average_quality,
            s=100,
            label=model,
        )


    axis.set_title(
        "Quality vs Latency"
    )

    axis.set_xlabel(
        "Average Latency (seconds)"
    )

    axis.set_ylabel(
        "Manual Quality Score (1–5)"
    )

    axis.set_ylim(
        0.5,
        5.5,
    )

    axis.legend(
        fontsize=8
    )

    axis.grid(
        True,
        alpha=0.25,
    )


    return save_chart(
        figure,
        timestamp,
        "05_quality_vs_latency",
    )


# ============================================================================
# CREATE ALL CHARTS
# ============================================================================

def create_charts(
    timestamp,
    rows,
):

    summary = build_summary(
        rows
    )

    models = list(
        dict.fromkeys(
            row["model"]
            for row in rows
        )
    )

    workloads = [
        workload["name"]
        for workload in WORKLOADS
    ]


    chart_paths = []


    chart_paths.append(
        chart_latency_vs_output_tokens(
            timestamp,
            summary,
            models,
            workloads,
        )
    )


    chart_paths.append(
        chart_tokens_per_second(
            timestamp,
            summary,
            models,
            workloads,
        )
    )


    chart_paths.append(
        chart_latency_vs_input_tokens(
            timestamp,
            summary,
            models,
            workloads,
        )
    )


    chart_paths.append(
        chart_latency_by_workload(
            timestamp,
            summary,
            models,
            workloads,
        )
    )


    quality_chart = (
        chart_quality_vs_latency(
            timestamp,
            rows,
        )
    )


    if quality_chart:

        chart_paths.append(
            quality_chart
        )


    return chart_paths


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    rows = []

    failures = []


    metadata = environment_metadata()


    print("=" * 80)

    print(
        "EXPERIMENT 01 — INFERENCE BASICS"
    )

    print("=" * 80)

    print(
        f"Device: {device_name()}"
    )

    print(
        f"Models: {len(MODEL_NAMES)}"
    )

    print(
        f"Runs per workload: {RUNS}"
    )


    for model_name in MODEL_NAMES:

        print(
            f"\nLoading: {model_name}"
        )


        try:

            tokenizer, model = (
                load_model(
                    model_name
                )
            )


            warmup(
                tokenizer,
                model
            )


            for workload in WORKLOADS:

                print(
                    f"  {workload['workload_id']} "
                    f"— {workload['name']}"
                )


                for run_number in range(
                    1,
                    RUNS + 1,
                ):

                    try:

                        result = (
                            generate_once(
                                tokenizer,
                                model,
                                workload[
                                    "prompt"
                                ],
                                workload[
                                    "max_new_tokens"
                                ],
                            )
                        )


                        rows.append({

                            "timestamp":
                                timestamp,

                            "model":
                                model_name,

                            "workload_id":
                                workload[
                                    "workload_id"
                                ],

                            "workload_name":
                                workload[
                                    "name"
                                ],

                            "run":
                                run_number,

                            "input_tokens":
                                result[
                                    "input_tokens"
                                ],

                            "output_tokens":
                                result[
                                    "output_tokens"
                                ],

                            "latency_seconds":
                                round(
                                    result[
                                        "latency_seconds"
                                    ],
                                    6,
                                ),

                            "output_tokens_per_second":
                                round(
                                    result[
                                        "output_tokens_per_second"
                                    ],
                                    4,
                                ),

                            "quality_score_manual":
                                "",

                            "output":
                                result[
                                    "output"
                                ].replace(
                                    "\n",
                                    " ",
                                ),

                            "device":
                                device_name(),

                            "pytorch_version":
                                torch.__version__,

                            "transformers_version":
                                transformers.__version__,

                            "error":
                                "",
                        })


                    except Exception as exc:

                        rows.append({

                            "timestamp":
                                timestamp,

                            "model":
                                model_name,

                            "workload_id":
                                workload[
                                    "workload_id"
                                ],

                            "workload_name":
                                workload[
                                    "name"
                                ],

                            "run":
                                run_number,

                            "input_tokens":
                                "",

                            "output_tokens":
                                "",

                            "latency_seconds":
                                "",

                            "output_tokens_per_second":
                                "",

                            "quality_score_manual":
                                "",

                            "output":
                                "",

                            "device":
                                device_name(),

                            "pytorch_version":
                                torch.__version__,

                            "transformers_version":
                                transformers.__version__,

                            "error":
                                (
                                    f"{type(exc).__name__}: "
                                    f"{exc}"
                                ),
                        })


            del model

            del tokenizer

            gc.collect()


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


        except Exception as exc:

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"  FAILED: {error}"
            )

            failures.append(
                (
                    model_name,
                    error,
                )
            )


    if not rows:

        raise RuntimeError(
            "No experiment results were produced."
        )


    csv_path, txt_path = (
        save_results(
            timestamp,
            rows,
            metadata,
            failures,
        )
    )


    chart_paths = create_charts(
        timestamp,
        rows,
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )


    print(
        f"CSV: {csv_path}"
    )

    print(
        f"TXT: {txt_path}"
    )


    print(
        "\nCharts:"
    )


    for chart in chart_paths:

        print(
            f"  {chart}"
        )


    if failures:

        print(
            "\nModels that could not be run:"
        )

        for model, error in failures:

            print(
                f"  {model}: {error}"
            )


    print(
        "\nNext step:"
    )

    print(
        "Review the CSV outputs and manually "
        "populate quality_score_manual (1–5) "
        "if you want the quality-vs-latency chart."
    )


if __name__ == "__main__":

    main()