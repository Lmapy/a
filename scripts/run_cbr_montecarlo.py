"""Monte Carlo trade-reorder CI on a CBR scalp ledger.

  python3 scripts/run_cbr_montecarlo.py
      [--trades results/cbr_gold_scalp/trades.csv]
      [--n-runs 5000]
      [--initial-equity 50000]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from strategies.scalp.monte_carlo import monte_carlo_reorder, write_monte_carlo


@dataclass
class _StubTrade:
    """The MC function takes a list of "trade" objects with `r_result`
    and `pnl` attributes; we wrap CSV rows in that shape."""
    r_result: float
    pnl: float


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", default="results/cbr_gold_scalp/trades.csv")
    ap.add_argument("--n-runs", type=int, default=5000)
    ap.add_argument("--initial-equity", type=float, default=50_000.0)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)

    trades_path = ROOT / args.trades
    if not trades_path.exists():
        print(f"[error] trades CSV not found: {trades_path}", file=sys.stderr)
        return 2
    df = pd.read_csv(trades_path)
    if df.empty:
        print("[mc] trades CSV is empty; nothing to do")
        return 0
    trades = [_StubTrade(r_result=float(r["r_result"]),
                          pnl=float(r["pnl"]))
               for _, r in df.iterrows()]
    payload = monte_carlo_reorder(
        trades, n_runs=args.n_runs,
        initial_equity=args.initial_equity)
    out_dir = ROOT / (args.output_dir or trades_path.parent)
    p = write_monte_carlo(payload, out_dir)

    print()
    print("=== MONTE CARLO SUMMARY ===")
    print(f"  n_trades:    {payload['n_trades']}")
    print(f"  n_runs:      {payload['n_runs']}")
    print(f"  actual:      total_r={payload['actual']['total_r']}  "
          f"max_dd={payload['actual']['max_drawdown_r']}")
    boot = payload["bootstrap_total_r"]
    print(f"  total_r CI:  p05={boot['p05']}  p50={boot['p50']}  p95={boot['p95']}")
    boot = payload["bootstrap_max_drawdown_r"]
    print(f"  max_dd CI:   p05={boot['p05']}  p50={boot['p50']}  p95={boot['p95']}")
    print(f"  share negative total_r:        "
          f"{payload['share_runs_negative_total_r']}")
    print(f"  share dd worse than actual:    "
          f"{payload['share_runs_dd_worse_than_actual']}")
    print(f"  output:      {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
