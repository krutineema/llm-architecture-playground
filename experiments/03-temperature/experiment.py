"""
Temperature Experiment
======================

Goal:
    Explore how temperature affects variation in model outputs.

We keep the model, prompt and other settings constant and vary
only the temperature.

Each temperature is tested multiple times.

Runtime:
    Ollama

Default model:
    gemma3:1b
"""

from datetime import datetime
from pathlib import Path

# pyrefly: ignore [missing-import]
import ollama


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemma3:1b"
#MODEL = "gpt-oss:20b"
#MODEL = "qwen3.5"

RESULTS_DIR = Path(__file__).parent / "results"

PROMPTS = [
    """Give me 5 possible names for an AI assistant designed to help
employees find information inside a company.

Return only the 5 names, one per line.""",
    """Explain how AI works in a few words.""",
    """Write a short story about a robot.""",
    """Translate "The quick brown fox jumps over the lazy dog" to French."""
]

TEMPERATURES = [
    0.0,
    0.3,
    0.7,
    1.0,
]

RUNS_PER_TEMPERATURE = 5


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    temperature: float,
    model: str = MODEL,
) -> str:
    """Generate a response at a specific temperature."""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": temperature,
        },
    )

    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model: str = MODEL) -> Path:
    """Run the temperature experiment and save all results."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = model.replace(":", "_")
    result_file = RESULTS_DIR / f"results_{model_filename}_{timestamp}.md"

    with result_file.open("w", encoding="utf-8") as file:

        file.write("# Temperature Experiment Results\n\n")
        file.write(f"**Date:** {datetime.now().isoformat()}\n\n")
        file.write(f"**Model:** `{model}`\n\n")
        file.write(
            f"**Runs per temperature:** `{RUNS_PER_TEMPERATURE}`\n\n"
        )

        file.write("---\n\n")

        for prompt_idx, prompt in enumerate(PROMPTS, start=1):

            file.write(f"## Prompt {prompt_idx}\n\n")
            file.write("```text\n")
            file.write(prompt.strip())
            file.write("\n```\n\n")

            for temperature in TEMPERATURES:

                file.write(f"### Temperature: `{temperature}`\n\n")

                for run_number in range(1, RUNS_PER_TEMPERATURE + 1):

                    response = generate(
                        prompt=prompt,
                        temperature=temperature,
                        model=model,
                    )

                    file.write(f"#### Run {run_number}\n\n")
                    file.write("```text\n")
                    file.write(response.strip())
                    file.write("\n```\n\n")

                file.write("---\n\n")

            file.write("---\n\n")

    return result_file


if __name__ == "__main__":
    result_file = run_experiment()

    print(f"Results saved to: {result_file}")