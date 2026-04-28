"""Canonical Dukascopy-only loader.

Public API:

    load_candles(symbol="XAUUSD", timeframe="M15",
                 source="dukascopy", years=None) -> pd.DataFrame

The returned DataFrame has the canonical schema:

    time            tz-aware UTC
    open  high  low  close
    volume          tick_volume sum (lots)
    spread_mean     mean (ask-bid) over the bar
    spread_max      max  (ask-bid) over the bar
    tick_count      number of ticks in the bar
    dataset_source  always "dukascopy"

Anything that doesn't pass the dataset_source check is rejected. The
old broker files (XAUUSD_H4_long.csv, _matched.csv) are deprecated and
moved to data/_deprecated_/; they cannot be loaded through this API.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DUKA_DIR = DATA_DIR / "dukascopy" / "candles"

CANONICAL_COLS = [
    "time", "open", "high", "low", "close",
    "volume", "spread_mean", "spread_max", "tick_count", "dataset_source",
]


def load_candles(symbol: str = "XAUUSD",
                 timeframe: str = "M15",
                 source: str = "dukascopy",
                 years: list[int] | None = None) -> pd.DataFrame:
    if source != "dukascopy":
        raise ValueError(
            f"Active pipeline accepts source='dukascopy' only "
            f"(got source={source!r}). The previous broker datasets "
            "have been deprecated; see data/_deprecated_/.")
    tf_dir = DUKA_DIR / symbol / timeframe
    if not tf_dir.exists():
        raise FileNotFoundError(
            f"No Dukascopy candles for {symbol}/{timeframe} on disk. "
            f"Expected directory: {tf_dir}\n"
            "Run `python3 scripts/fetch_dukascopy.py "
            "--symbol XAUUSD --start 2008-01-01 --end 2026-04-29` "
            "from an environment with internet access to "
            "datafeed.dukascopy.com.")
    # Read per-year files (parquet preferred, CSV fallback). The build
    # step writes year=YYYY.parquet; legacy local fixtures may have CSVs.
    parquet_files = sorted(tf_dir.glob("year=*.parquet"))
    csv_files     = sorted(tf_dir.glob("year=*.csv"))
    files = parquet_files or csv_files
    if years is not None:
        ext = "parquet" if parquet_files else "csv"
        wanted = {f"year={y}.{ext}" for y in years}
        files = [f for f in files if f.name in wanted]
    if not files:
        # last resort: a single merged file (used when per-year split
        # was not produced, e.g. a fetch_dukascopy.py local run).
        merged = list(tf_dir.glob("*.parquet"))
        if merged:
            files = [sorted(merged)[0]]
    if not files:
        raise FileNotFoundError(
            f"No year files in {tf_dir}. Run scripts/fetch_dukascopy.py "
            "or pull from the data-dukascopy sidecar branch.")
    frames = []
    for f in files:
        if f.suffix == ".parquet":
            frames.append(pd.read_parquet(f))
        else:
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "dataset_source" not in df.columns:
        raise ValueError(
            f"Candle file in {tf_dir} is missing the dataset_source "
            "column — refusing to load (active pipeline requires "
            "dataset_source == 'dukascopy').")
    bad = df[df["dataset_source"] != "dukascopy"]
    if len(bad):
        raise ValueError(
            f"{len(bad)} rows in {tf_dir} have dataset_source != "
            "'dukascopy'. The active pipeline does not accept mixed "
            "sources. Move the file to data/_deprecated_/ or refetch.")
    df = df[CANONICAL_COLS]
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Returns a Dukascopy-only frame dict {h4_long, h4, m15} compatible
    with the older callers. h4_long and h4 are the same series under
    Dukascopy (single-source); the matched/long distinction was a
    legacy artefact of having to mix two brokers."""
    h4 = load_candles(symbol="XAUUSD", timeframe="H4")
    m15 = load_candles(symbol="XAUUSD", timeframe="M15")
    return {"h4_long": h4, "h4": h4, "m15": m15}


def list_available() -> dict[str, list[int]]:
    """For diagnostics: which timeframes/years are on disk."""
    out: dict[str, list[int]] = {}
    base = DUKA_DIR / "XAUUSD"
    if not base.exists():
        return out
    for tf_dir in sorted(base.iterdir()):
        if not tf_dir.is_dir():
            continue
        years = []
        for f in sorted(list(tf_dir.glob("year=*.parquet"))
                        + list(tf_dir.glob("year=*.csv"))):
            try:
                years.append(int(f.stem.split("=")[1]))
            except (IndexError, ValueError):
                continue
        # dedupe + sort
        out[tf_dir.name] = sorted(set(years))
    return out
