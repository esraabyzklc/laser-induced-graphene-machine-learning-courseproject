"""Similarity analysis between LIG experimental samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.spatial.distance import cityblock, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from data_utils import TARGET_COLUMN, clean_lig_data, ensure_columns, load_lig_data

DEFAULT_FEATURES = ["Power", "Focus(cm)", "Speed(mm/s)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find experiments similar to a target sample.")
    parser.add_argument("--input", required=True, help="Path to dataset.")
    parser.add_argument("--output", default="results/tables/similarity_results.csv", help="Output CSV path.")
    parser.add_argument("--target-index", type=int, default=0, help="Index of the target experiment.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of most similar experiments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean_lig_data(load_lig_data(args.input))
    ensure_columns(df, DEFAULT_FEATURES)

    if args.target_index < 0 or args.target_index >= len(df):
        raise IndexError(f"target-index must be between 0 and {len(df) - 1}")

    X = df[DEFAULT_FEATURES]
    X_scaled = StandardScaler().fit_transform(X)
    target = X_scaled[args.target_index]

    cosine_scores = cosine_similarity([target], X_scaled)[0]
    euclidean_distances = [euclidean(row, target) for row in X_scaled]
    manhattan_distances = [cityblock(row, target) for row in X_scaled]

    results = df.copy()
    results["Cosine_Similarity"] = cosine_scores
    results["Euclidean_Distance"] = euclidean_distances
    results["Manhattan_Distance"] = manhattan_distances
    results = results.drop(index=args.target_index)
    results = results.sort_values("Cosine_Similarity", ascending=False).head(args.top_n)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    display_cols = DEFAULT_FEATURES + [col for col in [TARGET_COLUMN, "Cosine_Similarity", "Euclidean_Distance", "Manhattan_Distance"] if col in results.columns]
    print("Target experiment:")
    print(df.loc[args.target_index, DEFAULT_FEATURES + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])])
    print("\nMost similar experiments:")
    print(results[display_cols])
    print(f"\nSaved similarity table to: {output}")


if __name__ == "__main__":
    main()
