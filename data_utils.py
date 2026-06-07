"""Utility functions for Laser-Induced Graphene data analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


NUMERIC_COLUMNS = [
    "Power",
    "Focus(cm)",
    "Speed(mm/s)",
    "Frequency(kHz)",
    "Line Distance(mm)",
    "Pulse Width (ns)",
    "Resistance(Ohm)",
]

FEATURE_COLUMNS = [
    "Power",
    "Focus(cm)",
    "Speed(mm/s)",
    "Frequency(kHz)",
    "Line Distance(mm)",
    "Pulse Width (ns)",
]

TARGET_COLUMN = "Resistance(Ohm)"
CONDITION_COLUMN = "Condition"


def load_lig_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel LIG dataset with basic delimiter handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    # Most project files used either comma or semicolon delimiters.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, delimiter=";")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common spacing/name variations from the project files."""
    rename_map = {
        "Focus (cm)": "Focus(cm)",
        "Resistance (Ohm)": "Resistance(Ohm)",
        "Pulse Width(ns)": "Pulse Width (ns)",
        "Line Distance (mm)": "Line Distance(mm)",
        "Frequency (kHz)": "Frequency(kHz)",
        "Speed (mm/s)": "Speed(mm/s)",
    }
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df.rename(columns=rename_map)


def convert_numeric_columns(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    """Convert numeric columns that may contain comma decimals."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_lig_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common cleaning operations used across the project."""
    df = normalize_column_names(df)
    df = convert_numeric_columns(df)

    if CONDITION_COLUMN in df.columns:
        df[CONDITION_COLUMN] = df[CONDITION_COLUMN].replace({
            "TRUE": 1,
            "True": 1,
            True: 1,
            "FALSE": 0,
            "False": 0,
            False: 0,
        })
        df[CONDITION_COLUMN] = pd.to_numeric(df[CONDITION_COLUMN], errors="ignore")

    required = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col in df.columns]
    if required:
        df = df.dropna(subset=required)

    return df.reset_index(drop=True)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
