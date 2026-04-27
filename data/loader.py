"""Read OHLC CSVs from data/ as tz-aware UTC DataFrames.

This is the only place v2 code reads bar data from disk. Every other
module receives DataFrames produced here, so format normalisation is
done once.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

REQUIRED_COLS = ("time", "open", "high", "low", "close")
OPTIONAL_COLS = ("volume", "spread")


def load(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"missing data file: {p}")
    df = pd.read_csv(p)
    if "time" not in df.columns:
        raise ValueError(f"{name}: no 'time' column")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"{name}: missing required column {col}")
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[list(REQUIRED_COLS) + list(OPTIONAL_COLS)]
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "h4_long": load("XAUUSD_H4_long.csv"),
        "h4": load("XAUUSD_H4_matched.csv"),
        "m15": load("XAUUSD_M15_matched.csv"),
    }
