"""Build candles from dukascopy-node CSV exports.

dukascopy-node (https://github.com/Leo4815162342/dukascopy-node) is a
battle-tested Node CLI that handles Dukascopy's per-IP rate limiting.
We let it do the network step (fetching M1 bid + ask separately), then
this script combines them into mid-price OHLC + per-bar spread and
resamples to every higher timeframe.

Inputs (default --input-dir layout):

    <input-dir>/bid/*.csv       -- one CSV per file dukascopy-node emits
    <input-dir>/ask/*.csv

Each CSV row (M1 timeframe, --format csv):

    timestamp,open,high,low,close,volume     <- timestamp in ms epoch UTC

Outputs:

    <output-root>/<SYMBOL>/<TF>/year=YYYY.parquet  for TF in
        M1 M3 M5 M15 M30 H1 H4 D1
    <output-root>/<SYMBOL>/<TF>/<SYMBOL>_<TF>.parquet  (merged)
    <output-root>/<SYMBOL>/manifest.json

Every row carries dataset_source = "dukascopy".
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._dukascopy_codec import CANDLE_COLS, TIMEFRAMES_MIN


def _read_dn_csv(directory: Path) -> pd.DataFrame:
    """Concat every CSV under `directory` into one DataFrame.

    dukascopy-node M1 CSV has columns: timestamp,open,high,low,close,volume.
    Tolerant of partial-failure artefacts: 0-byte files and files that
    fail to parse (EmptyDataError) are SKIPPED with a warning rather
    than aborting. The downstream join handles missing years naturally
    (only timestamps present in BOTH bid and ask survive).
    """
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs found under {directory}")
    frames = []
    skipped: list[str] = []
    for f in files:
        size = f.stat().st_size
        if size == 0:
            skipped.append(f"{f.name} (0 bytes)")
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            skipped.append(f"{f.name} (no parseable columns)")
            continue
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" not in df.columns:
            skipped.append(f"{f.name} (missing timestamp column)")
            continue
        frames.append(df)
    if skipped:
        print(f"[build] WARN skipped {len(skipped)} file(s) under "
              f"{directory.name}/: {skipped}", file=sys.stderr)
    if not frames:
        raise SystemExit(f"no usable CSVs in {directory} after skip-filter")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _to_canonical(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Inner-join bid + ask on timestamp; emit mid-price OHLC + spread."""
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    bid = bid[cols].rename(columns={c: f"{c}_bid" for c in cols if c != "timestamp"})
    ask = ask[cols].rename(columns={c: f"{c}_ask" for c in cols if c != "timestamp"})
    df = pd.merge(bid, ask, on="timestamp", how="inner")
    if df.empty:
        raise SystemExit("bid/ask join is empty -- bid and ask CSVs do not "
                         "cover overlapping timestamps")
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open"]  = (df["open_bid"]  + df["open_ask"])  / 2.0
    df["high"]  = (df["high_bid"]  + df["high_ask"])  / 2.0
    df["low"]   = (df["low_bid"]   + df["low_ask"])   / 2.0
    df["close"] = (df["close_bid"] + df["close_ask"]) / 2.0
    df["volume"] = df["volume_bid"] + df["volume_ask"]
    df["spread_mean"] = df["close_ask"] - df["close_bid"]
    df["spread_max"]  = (df["high_ask"] - df["low_bid"]).clip(lower=0)
    df["tick_count"]  = 1   # M1 aggregation; finer detail not available from candle API
    df["dataset_source"] = "dukascopy"
    return df[CANDLE_COLS].copy()


def _resample(m1: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    if m1.empty or freq_minutes == 1:
        return m1.copy() if freq_minutes == 1 else pd.DataFrame(columns=CANDLE_COLS)
    df = m1.set_index("time")
    freq = f"{freq_minutes}min"
    out = df.resample(freq, closed="left", label="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        spread_mean=("spread_mean", "mean"),
        spread_max=("spread_max", "max"),
        tick_count=("tick_count", "sum"),
    ).dropna(subset=["open"]).reset_index()
    out["dataset_source"] = "dukascopy"
    return out[CANDLE_COLS]


def _write_per_year(df: pd.DataFrame, tf_dir: Path, symbol: str, tf: str) -> int:
    if df.empty:
        return 0
    tf_dir.mkdir(parents=True, exist_ok=True)
    # merged
    merged = tf_dir / f"{symbol}_{tf}.parquet"
    df.to_parquet(merged, index=False)
    # per-year
    df = df.copy()
    df["_y"] = df["time"].dt.year
    for year, g in df.groupby("_y"):
        g.drop(columns=["_y"]).to_parquet(tf_dir / f"year={int(year)}.parquet",
                                          index=False)
    return len(df)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--input-dir", required=True,
                   help="directory containing bid/ and ask/ subdirs of "
                        "dukascopy-node M1 CSVs")
    p.add_argument("--output-dir", required=True,
                   help="output root, e.g. output/dukascopy")
    p.add_argument("--start", required=True, help="UTC YYYY-MM-DD (for manifest)")
    p.add_argument("--end",   required=True, help="UTC YYYY-MM-DD (for manifest)")
    args = p.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    sym_dir = out_root / args.symbol

    print(f"[build] reading {in_root / 'bid'}", flush=True)
    bid = _read_dn_csv(in_root / "bid")
    print(f"        bid rows: {len(bid):,}", flush=True)
    print(f"[build] reading {in_root / 'ask'}", flush=True)
    ask = _read_dn_csv(in_root / "ask")
    print(f"        ask rows: {len(ask):,}", flush=True)

    m1 = _to_canonical(bid, ask)
    print(f"[build] mid M1 rows: {len(m1):,}  "
          f"first={m1['time'].iloc[0]}  last={m1['time'].iloc[-1]}", flush=True)

    # sanity warning on price range
    sample = float(m1["close"].iloc[0])
    if not (200.0 <= sample <= 10000.0):
        print(f"[build] WARN sample close {sample} outside expected gold "
              f"range [200, 10000]", file=sys.stderr)

    rows_per_tf: dict[str, int] = {}
    for tf, minutes in TIMEFRAMES_MIN.items():
        bars = m1 if minutes == 1 else _resample(m1, minutes)
        n = _write_per_year(bars, sym_dir / tf, args.symbol, tf)
        rows_per_tf[tf] = n
        print(f"  {tf}: {n:,} bars", flush=True)

    manifest = {
        "official_source": "dukascopy",
        "downloaded_via": "dukascopy-node",
        "symbol": args.symbol,
        "timeframes": list(TIMEFRAMES_MIN.keys()),
        "rows_per_timeframe": rows_per_tf,
        "range_start_utc": args.start,
        "range_end_utc": args.end,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": ("Single-source Dukascopy. Lower TFs derived from M1 mid "
                   "(bid+ask)/2; spread_mean per bar = ask_close - bid_close; "
                   "tick_count is aggregated from M1 (1 per minute, summed)."),
    }
    sym_dir.mkdir(parents=True, exist_ok=True)
    (sym_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build] manifest -> {sym_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
