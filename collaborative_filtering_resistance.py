"""Resistance prediction using similarity-weighted neighboring experiments."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from data_utils import TARGET_COLUMN, clean_lig_data, ensure_columns, load_lig_data

FEATURES = ["Power", "Focus(cm)", "Speed(mm/s)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict resistance using similar experiments.")
    parser.add_argument("--input", required=True, help="Path to dataset.")
    parser.add_argument("--target-index", type=int, default=0, help="Index of the target experiment.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of neighbors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean_lig_data(load_lig_data(args.input))
    ensure_columns(df, FEATURES + [TARGET_COLUMN])

    X_scaled = StandardScaler().fit_transform(df[FEATURES])
    similarity_matrix = cosine_similarity(X_scaled)
    similarity_scores = similarity_matrix[args.target_index]

    candidates = df.copy()
    candidates["Similarity"] = similarity_scores
    candidates = candidates.drop(index=args.target_index)
    neighbors = candidates.nlargest(args.top_n, "Similarity")

    weights = neighbors["Similarity"].clip(lower=0)
    if weights.sum() == 0:
        predicted_resistance = neighbors[TARGET_COLUMN].mean()
    else:
        predicted_resistance = np.average(neighbors[TARGET_COLUMN], weights=weights)

    print("Target experiment:")
    print(df.iloc[args.target_index])
    print("\nMost similar experiments:")
    print(neighbors[FEATURES + [TARGET_COLUMN, "Similarity"]])
    print(f"\nSimilarity-weighted predicted resistance: {predicted_resistance:.4f} Ohm")


if __name__ == "__main__":
    main()
