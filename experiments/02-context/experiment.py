"""
Context Experiment
==================

Goal:
    Explore how additional context changes an LLM's response.

The underlying task remains the same while we progressively add:
    1. No additional context
    2. Audience context
    3. Audience + constraints
    4. Conflicting context

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

#MODEL = "gemma3:1b"
#MODEL = "gpt-oss:20b"
MODEL = "qwen3.5"

RESULTS_DIR = Path(__file__).parent / "results"

BASE_TASK = """
Explain why a company might adopt AI.
"""

EXPERIMENTS = {
    "1_no_context": BASE_TASK,

    "2_audience": """
You are explaining this to a CFO who is skeptical about AI.

Explain why a company might adopt AI.
""",

    "3_audience_and_constraints": """
You are explaining this to a CFO who is skeptical about AI.

Explain why a company might adopt AI.

Requirements:
- Exactly 5 lines
- Avoid technical jargon
- Focus on measurable business outcomes
- Include one concrete example
""",

    "4_conflicting_context": """
You are writing for a highly technical audience.

However, avoid technical terminology and explain everything
as if the reader has never used software.

Explain why a company might adopt AI.
""",
}


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------

def generate(prompt: str, model: str = MODEL) -> str:
    """Generate a response from the selected model."""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model: str = MODEL) -> Path:
    """Run all context variations and save results to a file."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"results_{MODEL}_{timestamp}.md"

    with result_file.open("w", encoding="utf-8") as file:

        file.write("# Context Experiment Results\n\n")
        file.write(f"**Date:** {datetime.now().isoformat()}\n\n")
        file.write(f"**Model:** `{model}`\n\n")

        file.write("---\n\n")

        for name, prompt in EXPERIMENTS.items():

            file.write(f"## {name}\n\n")

            file.write("### Prompt\n\n")
            file.write("```text\n")
            file.write(prompt.strip())
            file.write("\n```\n\n")

            response = generate(prompt, model)

            file.write("### Response\n\n")
            file.write("```text\n")
            file.write(response.strip())
            file.write("\n```\n\n")

            file.write("---\n\n")

    return result_file


if __name__ == "__main__":
    result_file = run_experiment()

    print(f"Results saved to: {result_file}")