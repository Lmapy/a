"""Decode .bi5 ticks under <input-dir>/raw/<SYMBOL>/ and resample to
candles for all 8 timeframes. Writes parquet (and optional CSV) to
<output-dir>/<SYMBOL>/<TF>/<SYMBOL>_<TF>.parquet.

To stay within CI memory limits the script processes one chunk at a
time (year by default; quarter or month also supported). Per-chunk
candles are written to <output-dir>/<SYMBOL>/<TF>/_chunks/ and merged
at the end.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._dukascopy_codec import (
    CANDLE_COLS, TIMEFRAMES_MIN, chunk_ranges, decode_bi5, iter_hours,
    load_symbol_spec, raw_path_for, resample_ticks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", required=True)
    p.add_argument("--end",   required=True)
    p.add_argument("--input-dir",  default="output/dukascopy")
    p.add_argument("--output-dir", default="output/dukascopy")
    p.add_argument("--write-csv", action="store_true")
    p.add_argument("--chunk-by", choices=["year", "quarter", "month"], default="year")
    p.add_argument("--timeframes", nargs="+",
                   default=list(TIMEFRAMES_MIN.keys()),
                   help="subset of M1 M3 M5 M15 M30 H1 H4 D1")
    return p.parse_args()


def _decode_chunk_to_ticks(symbol: str, raw_root: Path, start: datetime,
                            end: datetime, price_scale: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ts in iter_hours(start, end):
        path = raw_path_for(raw_root, symbol, ts)
        if not path.exists():
            continue
        body = path.read_bytes()
        if not body:
            continue
        frames.append(decode_bi5(body, ts, price_scale))
    if not frames:
        return pd.DataFrame(columns=[
            "time", "bid", "ask", "mid", "spread",
            "bid_volume", "ask_volume", "volume",
        ])
    return pd.concat(frames, ignore_index=True).sort_values("time")


def _write_chunk(symbol: str, tf: str, freq_min: int, out_root: Path,
                  ticks: pd.DataFrame, chunk_label: str) -> tuple[int, float]:
    bars = resample_ticks(ticks, freq_min)
    if bars.empty:
        return 0, float("nan")
    chunk_dir = out_root / symbol / tf / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    p = chunk_dir / f"{symbol}_{tf}_{chunk_label}.parquet"
    bars.to_parquet(p, index=False)
    return len(bars), float(bars["close"].iloc[-1])


def _merge_chunks(symbol: str, tf: str, out_root: Path, write_csv: bool) -> int:
    """Merge per-chunk parquet into:

      - one merged file <out>/<SYM>/<TF>/<SYM>_<TF>.parquet
      - per-year files <out>/<SYM>/<TF>/year=YYYY.parquet

    The per-year split exists so the sidecar-branch commit step can push
    files that fit under GitHub's 100 MB per-file limit (M1 merged is
    ~250 MB for 6 years; per-year M1 is ~40 MB).
    """
    tf_dir = out_root / symbol / tf
    chunk_dir = tf_dir / "_chunks"
    if not chunk_dir.exists():
        return 0
    parts = sorted(chunk_dir.glob(f"{symbol}_{tf}_*.parquet"))
    if not parts:
        return 0
    frames = [pd.read_parquet(p) for p in parts]
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True))
    df = df[CANDLE_COLS]

    # 1) merged file (convenient for local use; may be too big to git-push)
    out_pq = tf_dir / f"{symbol}_{tf}.parquet"
    df.to_parquet(out_pq, index=False)
    if write_csv:
        df.to_csv(tf_dir / f"{symbol}_{tf}.csv", index=False)

    # 2) per-year files (always written; safe to commit to a sidecar branch)
    df["_year"] = df["time"].dt.year
    for year, g in df.groupby("_year"):
        ypath = tf_dir / f"year={int(year)}.parquet"
        g.drop(columns=["_year"]).to_parquet(ypath, index=False)
    return len(df)


def main() -> int:
    args = parse_args()
    spec = load_symbol_spec(args.symbol)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    raw_root = Path(args.input_dir) / "raw"
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    timeframes = [tf for tf in args.timeframes if tf in TIMEFRAMES_MIN]
    print(f"[build] {args.symbol}  {start.date()} -> {end.date()}  "
          f"chunk_by={args.chunk_by}  timeframes={timeframes}", flush=True)

    sanity_warned = False
    for chunk_start, chunk_end in chunk_ranges(start, end, args.chunk_by):
        label = f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"
        ticks = _decode_chunk_to_ticks(args.symbol, raw_root, chunk_start,
                                        chunk_end, spec.price_scale)
        if ticks.empty:
            print(f"  chunk {label}: no ticks", flush=True)
            continue
        # one-shot price sanity warning on the first non-empty chunk
        if not sanity_warned:
            sample_mid = float(ticks["mid"].iloc[0])
            msg = spec.sanity_warning(sample_mid)
            if msg:
                warnings.warn(f"[build] sanity: {msg}", RuntimeWarning)
            else:
                print(f"  sanity ok: first mid = {sample_mid:.4f} "
                      f"(in [{spec.expected_price_min}, "
                      f"{spec.expected_price_max}])", flush=True)
            sanity_warned = True

        for tf in timeframes:
            n, last = _write_chunk(args.symbol, tf, TIMEFRAMES_MIN[tf],
                                    out_root, ticks, label)
            print(f"  chunk {label} {tf}: {n:,} bars  last_close={last}",
                  flush=True)

    print("[build] merging chunks ...", flush=True)
    rows_per_tf: dict[str, int] = {}
    for tf in timeframes:
        rows = _merge_chunks(args.symbol, tf, out_root, args.write_csv)
        rows_per_tf[tf] = rows
        print(f"  {tf}: {rows:,} bars  -> "
              f"{out_root / args.symbol / tf / f'{args.symbol}_{tf}.parquet'}",
              flush=True)

    # remove _chunks dirs once merged
    for tf in timeframes:
        chunk_dir = out_root / args.symbol / tf / "_chunks"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)

    print(f"[build] done. rows per timeframe: {rows_per_tf}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
