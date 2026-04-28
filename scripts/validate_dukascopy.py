"""Validate Dukascopy candle parquet files.

For each <input-dir>/<SYMBOL>/<TF>/<SYMBOL>_<TF>.parquet check:

  1. monotonic timestamps
  2. no duplicate timestamps
  3. timestamps tz-aware UTC
  4. OHLC consistency (low <= open,close <= high; low <= high)
  5. no negative prices
  6. spread_mean >= 0, spread_max >= 0
  7. tick_count > 0
  8. correct timeframe boundary alignment
  9. missing-candle summary (expected n bars between [first, last] given
     <freq>; report missing windows but DO NOT INTERPOLATE)
 10. dataset_source == "dukascopy" on every row

Writes <input-dir>/<SYMBOL>/validation_report.json. Exits 1 if any
hard rule fails (1-7, 10). Missing candles + boundary misalignment
are warnings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._dukascopy_codec import TIMEFRAMES_MIN


def _aligned(ts: pd.Series, minutes: int) -> pd.Series:
    if minutes == 1:
        return ts.dt.second.eq(0)
    if minutes < 60:
        return ts.dt.second.eq(0) & (ts.dt.minute % minutes).eq(0)
    if minutes == 60:
        return ts.dt.second.eq(0) & ts.dt.minute.eq(0)
    if minutes == 240:
        return (ts.dt.second.eq(0) & ts.dt.minute.eq(0)
                & (ts.dt.hour % 4).eq(0))
    if minutes == 1440:
        return (ts.dt.second.eq(0) & ts.dt.minute.eq(0)
                & ts.dt.hour.eq(0))
    return pd.Series(False, index=ts.index)


def validate_one(parquet: Path, tf: str) -> dict:
    minutes = TIMEFRAMES_MIN[tf]
    out: dict = {"timeframe": tf, "path": str(parquet),
                 "hard_failures": [], "warnings": [], "rows": 0}
    if not parquet.exists():
        out["hard_failures"].append("file_missing")
        return out
    df = pd.read_parquet(parquet)
    out["rows"] = int(len(df))
    if df.empty:
        out["hard_failures"].append("empty")
        return out

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    out["first_time"] = str(df["time"].iloc[0])
    out["last_time"]  = str(df["time"].iloc[-1])

    # 1. monotonic
    if not df["time"].is_monotonic_increasing:
        out["hard_failures"].append("non_monotonic_timestamps")
    # 2. duplicates
    n_dup = int(df["time"].duplicated().sum())
    if n_dup:
        out["hard_failures"].append(f"duplicate_timestamps:{n_dup}")
    # 3. UTC
    tz = getattr(df["time"].dt, "tz", None)
    if tz is None or str(tz) != "UTC":
        out["hard_failures"].append(f"non_utc_timestamps:{tz}")
    # 4. OHLC consistency
    bad = df[(df["low"] > df[["open", "close"]].min(axis=1))
             | (df["high"] < df[["open", "close"]].max(axis=1))
             | (df["low"] > df["high"])]
    if len(bad):
        out["hard_failures"].append(f"ohlc_inconsistent:{len(bad)}")
    # 5. negative prices
    neg = (df[["open", "high", "low", "close"]] < 0).any(axis=1).sum()
    if neg:
        out["hard_failures"].append(f"negative_prices:{int(neg)}")
    # 6. spread non-negative
    if (df["spread_mean"] < 0).any():
        out["hard_failures"].append("negative_spread_mean")
    if (df["spread_max"] < 0).any():
        out["hard_failures"].append("negative_spread_max")
    # 7. tick_count > 0
    if (df["tick_count"] <= 0).any():
        out["hard_failures"].append("tick_count_not_positive")
    # 10. dataset_source
    if "dataset_source" not in df.columns:
        out["hard_failures"].append("missing_dataset_source_column")
    elif (df["dataset_source"] != "dukascopy").any():
        out["hard_failures"].append("dataset_source_not_dukascopy")

    # 8. boundary alignment
    n_misaligned = int((~_aligned(df["time"], minutes)).sum())
    if n_misaligned:
        out["warnings"].append(f"misaligned_to_{minutes}min:{n_misaligned}")

    # 9. missing-candle summary (expected vs actual bar count, ignoring
    # weekends — gaps > 1 bar are reported but not interpolated)
    diffs = df["time"].diff().dt.total_seconds().div(60).iloc[1:]
    gap_bars = diffs[diffs > minutes]
    out["gap_summary"] = {
        "n_gaps": int(len(gap_bars)),
        "max_gap_min": float(gap_bars.max()) if len(gap_bars) else 0.0,
        "n_large_gaps_over_6_bars": int((gap_bars > minutes * 6).sum()),
    }

    # numeric quality summary
    out["price_range"] = {
        "min_low":  float(df["low"].min()),
        "max_high": float(df["high"].max()),
        "median_close": float(df["close"].median()),
    }
    out["spread_range"] = {
        "spread_mean_avg": float(df["spread_mean"].mean()),
        "spread_max_max": float(df["spread_max"].max()),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--input-dir", default="output/dukascopy")
    args = p.parse_args()

    in_root = Path(args.input_dir) / args.symbol
    if not in_root.exists():
        print(f"no candles dir at {in_root}", file=sys.stderr)
        return 1

    report = {"symbol": args.symbol, "timeframes": {}}
    overall_hard_failures = 0
    for tf in TIMEFRAMES_MIN:
        pq = in_root / tf / f"{args.symbol}_{tf}.parquet"
        result = validate_one(pq, tf)
        report["timeframes"][tf] = result
        overall_hard_failures += len(result["hard_failures"])
        status = "FAIL" if result["hard_failures"] else "ok"
        print(f"[validate] {tf}: {status}  rows={result['rows']:,}  "
              f"hard={len(result['hard_failures'])}  "
              f"warn={len(result.get('warnings', []))}", flush=True)
        if result["hard_failures"]:
            for fh in result["hard_failures"]:
                print(f"    HARD FAIL: {fh}", flush=True)

    report["overall_hard_failures"] = overall_hard_failures
    out = Path(args.input_dir) / args.symbol / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[validate] report: {out}", flush=True)
    return 0 if overall_hard_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
