"""Naive Bayes-style condition prediction for numerical LIG features."""

from __future__ import annotations

import argparse
from math import sqrt

import numpy as np

from data_utils import CONDITION_COLUMN, FEATURE_COLUMNS, clean_lig_data, ensure_columns, load_lig_data


def gaussian_probability(values, x: float) -> float:
    mean = values.mean()
    std = values.std()
    if std == 0 or np.isnan(std):
        return 1.0 if np.isclose(x, mean) else 1e-12
    return (1 / (std * sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def class_probability(df_class, full_size: int, criteria: dict[str, float]) -> float:
    probability = len(df_class) / full_size
    for feature, value in criteria.items():
        probability *= gaussian_probability(df_class[feature], value)
    return probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Condition using a Gaussian Naive Bayes-style calculation.")
    parser.add_argument("--input", required=True, help="Path to dataset.")
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--focus", type=float, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--frequency", type=float, required=True)
    parser.add_argument("--line-distance", type=float, required=True)
    parser.add_argument("--pulse-width", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean_lig_data(load_lig_data(args.input))
    ensure_columns(df, FEATURE_COLUMNS + [CONDITION_COLUMN])

    criteria = {
        "Power": args.power,
        "Focus(cm)": args.focus,
        "Speed(mm/s)": args.speed,
        "Frequency(kHz)": args.frequency,
        "Line Distance(mm)": args.line_distance,
        "Pulse Width (ns)": args.pulse_width,
    }

    probabilities = {}
    for condition_value, subset in df.groupby(CONDITION_COLUMN):
        probabilities[condition_value] = class_probability(subset, len(df), criteria)

    predicted = max(probabilities, key=probabilities.get)
    print("Class probabilities:")
    for label, probability in probabilities.items():
        print(f"Condition={label}: {probability:.6e}")
    print(f"Predicted Condition: {predicted}")


if __name__ == "__main__":
    main()
