"""Generate synthetic LIG samples with GPR resistance and predicted condition labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from data_utils import CONDITION_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN, clean_lig_data, ensure_columns, load_lig_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic LIG data with GPR and condition labels.")
    parser.add_argument("--input", required=True, help="Path to cleaned or raw dataset.")
    parser.add_argument("--output", default="data/synthetic/lig_synthetic_gpr_with_condition.csv", help="Output CSV path.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of synthetic samples.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    df = clean_lig_data(load_lig_data(args.input))
    ensure_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN, CONDITION_COLUMN])
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, CONDITION_COLUMN])

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    conditions = df[CONDITION_COLUMN].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-2,
        n_restarts_optimizer=10,
        random_state=args.random_state,
    )
    gpr.fit(X_scaled, y)

    condition_model = RandomForestClassifier(n_estimators=100, random_state=args.random_state, class_weight="balanced")
    condition_model.fit(X, conditions)

    X_synthetic = np.zeros((args.n_samples, len(FEATURE_COLUMNS)))
    for i, feature in enumerate(FEATURE_COLUMNS):
        X_synthetic[:, i] = rng.uniform(df[feature].min(), df[feature].max(), args.n_samples)

    X_synthetic_scaled = scaler.transform(X_synthetic)
    y_mean, y_std = gpr.predict(X_synthetic_scaled, return_std=True)
    condition_pred = condition_model.predict(X_synthetic)

    synthetic = pd.DataFrame(X_synthetic, columns=FEATURE_COLUMNS)
    synthetic[TARGET_COLUMN] = y_mean
    synthetic["Resistance_Uncertainty"] = y_std
    synthetic[CONDITION_COLUMN] = condition_pred
    synthetic["Data_Source"] = "synthetic_gpr"

    expanded = pd.concat([df.assign(Data_Source="experimental"), synthetic], ignore_index=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(output, index=False)

    print(f"Generated {len(synthetic)} synthetic samples and saved expanded dataset to: {output}")
    print(f"Optimized kernel: {gpr.kernel_}")


if __name__ == "__main__":
    main()
