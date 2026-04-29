"""Run a CBR scalp parameter sweep.

  python3 scripts/run_cbr_sweep.py
      [--base configs/cbr_gold_scalp.yaml]
      [--sweep configs/cbr_gold_scalp_sweep.yaml]
      [--start 2024-01-01]   [--end 2026-04-29]
      [--limit-bars 500000]  [--output-stem cbr_sweep]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.loader import load_candles
from strategies.scalp.config import CBRGoldScalpConfig
from strategies.scalp.sweep import run_sweep


def _load_dxy_safe() -> pd.DataFrame | None:
    try:
        return load_candles(symbol="DXY", timeframe="H1")
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",  default="configs/cbr_gold_scalp.yaml")
    ap.add_argument("--sweep", default="configs/cbr_gold_scalp_sweep.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end",   default=None)
    ap.add_argument("--limit-bars", type=int, default=None)
    ap.add_argument("--output-stem", default="cbr_sweep")
    args = ap.parse_args(argv)

    base_path = ROOT / args.base
    sweep_path = ROOT / args.sweep
    if not base_path.exists() or not sweep_path.exists():
        print("[error] missing config(s)", file=sys.stderr)
        return 2
    base = CBRGoldScalpConfig.from_yaml(base_path)
    if args.start: base.start_date = args.start
    if args.end:   base.end_date = args.end

    print("[1/3] loading data ...")
    m1 = load_candles(symbol=base.symbol, timeframe=base.primary_timeframe)
    h1 = load_candles(symbol=base.symbol, timeframe=base.bias_timeframe)
    if base.start_date:
        m1 = m1[m1["time"] >= pd.Timestamp(base.start_date, tz="UTC")]
        h1 = h1[h1["time"] >= pd.Timestamp(base.start_date, tz="UTC")]
    if base.end_date:
        m1 = m1[m1["time"] <  pd.Timestamp(base.end_date, tz="UTC")]
        h1 = h1[h1["time"] <  pd.Timestamp(base.end_date, tz="UTC")]
    if args.limit_bars:
        m1 = m1.tail(args.limit_bars).reset_index(drop=True)
        if not m1.empty:
            h1 = h1[(h1["time"] >= m1["time"].iloc[0] - pd.Timedelta(hours=2))
                       & (h1["time"] <= m1["time"].iloc[-1])]
    print(f"      M1: {len(m1):,}   H1: {len(h1):,}")
    dxy = _load_dxy_safe()

    out_dir = ROOT / "results" / args.output_stem
    print(f"[2/3] running sweep -> {out_dir}")
    summary = run_sweep(base_yaml=base_path, sweep_yaml=sweep_path,
                         m1=m1, h1=h1, dxy=dxy, output_dir=out_dir)

    print()
    print("=== SWEEP SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
