"""Generate synthetic LIG samples using Gaussian Process Regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from data_utils import FEATURE_COLUMNS, TARGET_COLUMN, clean_lig_data, ensure_columns, load_lig_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic LIG data with GPR.")
    parser.add_argument("--input", required=True, help="Path to cleaned or raw dataset.")
    parser.add_argument("--output", default="data/synthetic/lig_synthetic_gpr.csv", help="Output CSV path.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of synthetic samples.")
    parser.add_argument(
        "--sampling",
        choices=["uniform", "normal"],
        default="uniform",
        help="Sampling method for synthetic feature values.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def sample_features(df: pd.DataFrame, n_samples: int, method: str, rng: np.random.Generator) -> np.ndarray:
    X_synthetic = np.zeros((n_samples, len(FEATURE_COLUMNS)))
    for i, feature in enumerate(FEATURE_COLUMNS):
        if method == "uniform":
            X_synthetic[:, i] = rng.uniform(df[feature].min(), df[feature].max(), n_samples)
        else:
            X_synthetic[:, i] = rng.normal(df[feature].mean(), df[feature].std(), n_samples)
            X_synthetic[:, i] = np.clip(X_synthetic[:, i], df[feature].min(), df[feature].max())
    return X_synthetic


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    df = clean_lig_data(load_lig_data(args.input))
    ensure_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

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

    X_synthetic = sample_features(df, args.n_samples, args.sampling, rng)
    X_synthetic_scaled = scaler.transform(X_synthetic)
    y_mean, y_std = gpr.predict(X_synthetic_scaled, return_std=True)

    synthetic = pd.DataFrame(X_synthetic, columns=FEATURE_COLUMNS)
    synthetic[TARGET_COLUMN] = y_mean
    synthetic["Resistance_Uncertainty"] = y_std
    synthetic["Data_Source"] = "synthetic_gpr"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output, index=False)

    print(f"Generated {len(synthetic)} synthetic samples.")
    print(f"Saved to: {output}")
    print(f"Optimized kernel: {gpr.kernel_}")


if __name__ == "__main__":
    main()
