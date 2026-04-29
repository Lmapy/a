"""Run a single CBR-style gold scalp backtest from a YAML config.

  python3 scripts/run_cbr_backtest.py
      [--config configs/cbr_gold_scalp.yaml]
      [--start 2024-01-01]  [--end 2026-04-29]
      [--limit-bars 200000]    # cap for smoke runs
      [--output-stem mytest]   # writes results/mytest/{...}

Outputs land under `results/cbr_gold_scalp/` by default:

    trades.csv             one row per realised trade
    setups.csv             one row per setup (taken or skipped)
    summary.json           aggregate metrics + funnel + breakdowns
    config_used.json       the exact resolved config (reproducibility)
    validation_report.json data validation findings
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.loader import load_candles
from strategies.scalp.config import CBRGoldScalpConfig
from strategies.scalp.engine import run_backtest
from strategies.scalp.metrics import write_outputs


def _load_dxy_safe() -> pd.DataFrame | None:
    """Try to load DXY data. Returns None if not on disk."""
    try:
        return load_candles(symbol="DXY", timeframe="H1")
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cbr_gold_scalp.yaml")
    ap.add_argument("--start", default=None,
                    help="ISO start date (overrides config)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit-bars", type=int, default=None,
                    help="cap M1 bars for smoke runs (most recent N)")
    ap.add_argument("--output-stem", default=None,
                    help="overrides cfg.output_dir basename")
    args = ap.parse_args(argv)

    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        print(f"[error] config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = CBRGoldScalpConfig.from_yaml(cfg_path)
    if args.start: cfg.start_date = args.start
    if args.end:   cfg.end_date = args.end
    if args.output_stem:
        cfg.output_dir = f"results/{args.output_stem}"

    issues = cfg.validate()
    if issues:
        print("[config validation issues]")
        for x in issues:
            print(f"  - {x}")

    print(f"[1/4] loading data: {cfg.symbol} {cfg.primary_timeframe} + {cfg.bias_timeframe}")
    m1 = load_candles(symbol=cfg.symbol, timeframe=cfg.primary_timeframe)
    h1 = load_candles(symbol=cfg.symbol, timeframe=cfg.bias_timeframe)

    if cfg.start_date:
        m1 = m1[m1["time"] >= pd.Timestamp(cfg.start_date, tz="UTC")]
        h1 = h1[h1["time"] >= pd.Timestamp(cfg.start_date, tz="UTC")]
    if cfg.end_date:
        m1 = m1[m1["time"] <  pd.Timestamp(cfg.end_date,   tz="UTC")]
        h1 = h1[h1["time"] <  pd.Timestamp(cfg.end_date,   tz="UTC")]
    if args.limit_bars:
        m1 = m1.tail(args.limit_bars).reset_index(drop=True)
        # h1 reduced to the matching range
        if not m1.empty:
            h1 = h1[h1["time"] <= m1["time"].iloc[-1]]
            h1 = h1[h1["time"] >= (m1["time"].iloc[0] - pd.Timedelta(hours=2))]
            h1 = h1.reset_index(drop=True)

    print(f"      M1: {len(m1):,} bars   H1: {len(h1):,} bars")

    print(f"[2/4] loading optional DXY (mode={cfg.dxy.dxy_mode}) ...")
    dxy = _load_dxy_safe() if cfg.dxy.dxy_mode != "OFF" else None
    print(f"      dxy_available={dxy is not None and (dxy is not None and not dxy.empty)}")

    print("[3/4] running backtest ...")
    results = run_backtest(cfg, m1, h1, dxy=dxy)

    print("[4/4] writing outputs ...")
    out_dir = ROOT / cfg.output_dir
    summary = write_outputs(results=results, cfg=cfg, output_dir=out_dir)

    m = summary["metrics"]
    print()
    print("=== SUMMARY ===")
    print(f"runtime:        {summary['runtime_s']}s")
    print(f"M1 bars:        {summary['n_m1_bars']:,}")
    if "total_trades" in m and m["total_trades"]:
        print(f"trades:         {m['total_trades']}")
        print(f"win_rate:       {m['win_rate']:.2%}")
        print(f"avg R:          {m['avg_r']:.3f}")
        print(f"total R:        {m['total_r']:.2f}")
        print(f"profit factor:  {m['profit_factor']}")
        print(f"max DD (R):     {m['max_drawdown_r']:.2f}")
        print(f"biggest share:  {m['biggest_trade_share']:.2%}")
        print(f"max consec L:   {m['max_consecutive_losses']}")
    else:
        print(f"trades:         {m.get('total_trades', 0)} (see setups.csv for skip reasons)")
        print(f"setups logged:  {m.get('n_setups_logged', 0)}")
    print(f"output dir:     {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
