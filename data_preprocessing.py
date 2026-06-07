"""Clean the Laser-Induced Graphene dataset and save a processed CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import clean_lig_data, load_lig_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean LIG experimental dataset.")
    parser.add_argument("--input", required=True, help="Path to raw CSV/XLSX dataset.")
    parser.add_argument("--output", default="data/processed/lig_cleaned.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_lig_data(args.input)
    cleaned = clean_lig_data(df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)

    print(f"Saved cleaned dataset to: {output}")
    print(f"Shape: {cleaned.shape}")
    print(cleaned.head())


if __name__ == "__main__":
    main()
