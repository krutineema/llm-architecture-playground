from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).parent / "results"


def get_latest_results_file():
    csv_files = sorted(RESULTS_DIR.glob("tokenisation_results*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No tokenisation_results CSV files found in {RESULTS_DIR}")
    return csv_files[-1]


def main():

    results_file = get_latest_results_file()
    print(f"Reading results from: {results_file}")

    # Extract timestamp from filename or generate a new one
    match = re.search(r"(\d{8}_\d{6})", results_file.name)
    if match:
        timestamp = match.group(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(results_file)

    summary = (
        df.groupby(["input_type", "tokenizer"])["tokens"]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(
        index="input_type",
        columns="tokenizer",
        values="tokens",
    )

    ax = pivot.plot(kind="bar")

    ax.set_title("Token Count by Input Type and Tokenizer")
    ax.set_xlabel("Input Type")
    ax.set_ylabel("Token Count")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output = RESULTS_DIR / f"token_counts_{timestamp}.png"

    plt.savefig(output, dpi=150)

    print(f"Saved chart to {output}")


if __name__ == "__main__":
    main()