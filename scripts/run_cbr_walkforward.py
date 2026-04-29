"""Walk-forward / OOS split for the CBR scalp engine.

  python3 scripts/run_cbr_walkforward.py
      [--config configs/cbr_gold_scalp.yaml]
      [--train-fraction 0.70]
      [--start ISO]   [--end ISO]
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
from strategies.scalp.walk_forward import run_walk_forward


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cbr_gold_scalp.yaml")
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--output-stem", default=None)
    args = ap.parse_args(argv)

    cfg = CBRGoldScalpConfig.from_yaml(ROOT / args.config)
    if args.start: cfg.start_date = args.start
    if args.end:   cfg.end_date = args.end
    if args.output_stem:
        cfg.output_dir = f"results/{args.output_stem}"

    print("[1/2] loading data ...")
    m1 = load_candles(symbol=cfg.symbol, timeframe=cfg.primary_timeframe)
    h1 = load_candles(symbol=cfg.symbol, timeframe=cfg.bias_timeframe)
    if cfg.start_date:
        m1 = m1[m1["time"] >= pd.Timestamp(cfg.start_date, tz="UTC")]
        h1 = h1[h1["time"] >= pd.Timestamp(cfg.start_date, tz="UTC")]
    if cfg.end_date:
        m1 = m1[m1["time"] <  pd.Timestamp(cfg.end_date, tz="UTC")]
        h1 = h1[h1["time"] <  pd.Timestamp(cfg.end_date, tz="UTC")]
    m1 = m1.reset_index(drop=True); h1 = h1.reset_index(drop=True)
    print(f"      m1 rows {len(m1):,}, h1 rows {len(h1):,}")

    print(f"[2/2] running walk-forward (train_fraction={args.train_fraction}) ...")
    out_dir = ROOT / cfg.output_dir
    payload = run_walk_forward(
        cfg=cfg, m1=m1, h1=h1,
        train_fraction=args.train_fraction,
        output_dir=out_dir,
    )

    deg = payload["degradation"]
    print()
    print("=== WALK-FORWARD SUMMARY ===")
    print(f"  split at:     {payload['split_at']}")
    print(f"  IS  trades:   {deg['in_sample_trades']}     "
          f"OOS trades: {deg['out_of_sample_trades']}")
    print(f"  IS  expect:   {deg['expectancy_in_sample']:.3f}    "
          f"OOS expect: {deg['expectancy_out_of_sample']:.3f}")
    print(f"  expect ratio: {deg.get('expectancy_ratio')}")
    print(f"  pf ratio:     {deg.get('profit_factor_ratio')}")
    print(f"  dd ratio:     {deg.get('drawdown_ratio')}")
    print(f"  flags:        {deg['flags']}")
    print(f"  stable:       {deg['stable']}")
    print(f"  output:       {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
