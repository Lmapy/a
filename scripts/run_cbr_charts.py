"""Render per-trade debug charts from a CBR scalp trade ledger.

  python3 scripts/run_cbr_charts.py
      [--trades results/cbr_gold_scalp/trades.csv]
      [--max-charts 30]
      [--output-dir results/cbr_gold_scalp/charts]

Requires matplotlib. If not installed, prints a warning and exits 0
without writing anything (so the rest of the pipeline isn't blocked).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_candles
from strategies.scalp.charts import render_all_trades


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", default="results/cbr_gold_scalp/trades.csv")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--max-charts", type=int, default=30)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)

    trades_path = ROOT / args.trades
    if not trades_path.exists():
        print(f"[error] trades CSV not found: {trades_path}", file=sys.stderr)
        return 2
    out_dir = ROOT / (args.output_dir or trades_path.parent / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] loading m1 data for {args.symbol} ...")
    m1 = load_candles(symbol=args.symbol, timeframe="M1")

    print(f"[2/2] rendering up to {args.max_charts} per-trade charts ...")
    n = render_all_trades(trades_path, m1, out_dir, max_charts=args.max_charts)
    print(f"      rendered {n} chart(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
