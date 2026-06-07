"""Visualization routines for Laser-Induced Graphene experimental data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import FEATURE_COLUMNS, TARGET_COLUMN, clean_lig_data, ensure_columns, load_lig_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LIG data visualizations.")
    parser.add_argument("--input", required=True, help="Path to cleaned or raw dataset.")
    parser.add_argument("--figures-dir", default="results/figures", help="Directory for output figures.")
    return parser.parse_args()


def save_boxplots(df, figures_dir: Path) -> None:
    columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col in df.columns]
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(4 * len(columns), 6))
    if len(columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        df[column].plot(kind="box", ax=ax, patch_artist=True)
        ax.set_title(f"{column} Boxplot")
        ax.set_ylabel("Value")
    plt.tight_layout()
    fig.savefig(figures_dir / "boxplots.png", dpi=300)
    plt.close(fig)


def save_scatter_plots(df, figures_dir: Path) -> None:
    ensure_columns(df, ["Focus(cm)", "Power", TARGET_COLUMN])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    df.plot(kind="scatter", x="Focus(cm)", y=TARGET_COLUMN, ax=axes[0])
    df.plot(kind="scatter", x="Power", y=TARGET_COLUMN, ax=axes[1])
    axes[0].set_title("Focus vs Resistance")
    axes[1].set_title("Power vs Resistance")
    plt.tight_layout()
    fig.savefig(figures_dir / "scatter_resistance.png", dpi=300)
    plt.close(fig)


def save_bubble_plot(df, figures_dir: Path) -> None:
    ensure_columns(df, ["Power", "Focus(cm)", TARGET_COLUMN])
    sizes = df[TARGET_COLUMN].clip(lower=0) * 30
    colors = df["Condition"] if "Condition" in df.columns else df[TARGET_COLUMN]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["Power"],
        df["Focus(cm)"],
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.7,
        edgecolors="w",
        linewidth=0.5,
    )
    ax.set_title("Power vs Focus with Resistance Size")
    ax.set_xlabel("Power")
    ax.set_ylabel("Focus (cm)")
    fig.colorbar(scatter, ax=ax, label="Condition" if "Condition" in df.columns else TARGET_COLUMN)
    plt.tight_layout()
    fig.savefig(figures_dir / "bubble_power_focus.png", dpi=300)
    plt.close(fig)


def save_pairplot(df, figures_dir: Path) -> None:
    columns = [col for col in ["Power", "Focus(cm)", "Speed(mm/s)", TARGET_COLUMN] if col in df.columns]
    if len(columns) >= 2:
        plot = sns.pairplot(df[columns].dropna())
        plot.fig.suptitle("Pairwise Feature Relationships", y=1.02)
        plot.savefig(figures_dir / "pairplot_features.png", dpi=300)
        plt.close(plot.fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = clean_lig_data(load_lig_data(args.input))
    save_boxplots(df, figures_dir)
    save_scatter_plots(df, figures_dir)
    save_bubble_plot(df, figures_dir)
    save_pairplot(df, figures_dir)

    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
