"""Bayesian posterior probability analysis for LIG resistance ranges."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_utils import TARGET_COLUMN, clean_lig_data, ensure_columns, load_lig_data


def discretize_column(df: pd.DataFrame, column: str, bins: int, new_column: str) -> pd.DataFrame:
    df = df.copy()
    df[new_column] = pd.cut(df[column], bins=bins, labels=False, include_lowest=True)
    return df


def calculate_posterior(
    df: pd.DataFrame,
    feature_bin_column: str,
    target_bin_column: str,
    target_bins: list[int],
) -> pd.Series:
    """Calculate P(target bin in target_bins | feature bin)."""
    feature_prob = df[feature_bin_column].value_counts(normalize=True).sort_index()
    conditional_prob = (
        df.groupby(feature_bin_column)[target_bin_column]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .sort_index()
    )
    target_prob = df[target_bin_column].value_counts(normalize=True)

    available_bins = [b for b in target_bins if b in target_prob.index]
    if not available_bins:
        raise ValueError(f"None of the requested target bins exist: {target_bins}")

    numerator = conditional_prob.reindex(columns=available_bins, fill_value=0).sum(axis=1) * feature_prob
    denominator = target_prob.reindex(available_bins).sum()
    return numerator / denominator


def plot_posterior(posterior: pd.Series, target_label: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=posterior.index.astype(str), y=posterior.values, ax=ax, palette="viridis")
    ax.set_title(f"Posterior Probabilities for Resistance {target_label}")
    ax.set_xlabel("Power Bin")
    ax.set_ylabel("Posterior Probability")
    plt.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Bayesian resistance range analysis.")
    parser.add_argument("--input", required=True, help="Path to dataset.")
    parser.add_argument("--feature", default="Power", help="Feature column to condition on.")
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target resistance column.")
    parser.add_argument("--feature-bins", type=int, default=5, help="Number of feature bins.")
    parser.add_argument("--target-bins", type=int, default=5, help="Number of resistance bins.")
    parser.add_argument("--selected-target-bin", type=int, default=1, help="Target bin used for posterior analysis.")
    parser.add_argument("--figures-dir", default="results/figures", help="Directory for output figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean_lig_data(load_lig_data(args.input))
    ensure_columns(df, [args.feature, args.target])

    df = discretize_column(df, args.feature, args.feature_bins, "feature_bin")
    df = discretize_column(df, args.target, args.target_bins, "target_bin")
    posterior = calculate_posterior(df, "feature_bin", "target_bin", [args.selected_target_bin])

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / "bayesian_posterior_resistance.png"
    plot_posterior(posterior, f"in bin {args.selected_target_bin}", output)

    print("Posterior probabilities:")
    print(posterior)
    print(f"Figure saved to: {output}")


if __name__ == "__main__":
    main()
