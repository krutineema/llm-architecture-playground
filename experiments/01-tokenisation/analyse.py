from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS = (
    Path(__file__).parents[2]
    / "results"
    / "tokenisation_results.csv"
)


def main():

    df = pd.read_csv(RESULTS)

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

    output = RESULTS.parent / "token_counts.png"

    plt.savefig(output, dpi=150)

    print(f"Saved chart to {output}")


if __name__ == "__main__":
    main()