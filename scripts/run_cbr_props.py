"""Feed a CBR scalp trade ledger into the prop-firm passing engine.

  python3 scripts/run_cbr_props.py
      [--trades results/cbr_gold_scalp/trades.csv]
      [--account topstep_50k]
      [--instrument MGC]
      [--n-runs 500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.scalp.prop_bridge import run_prop_replay


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", default="results/cbr_gold_scalp/trades.csv")
    ap.add_argument("--account", default="topstep_50k")
    ap.add_argument("--instrument", default="MGC")
    ap.add_argument("--n-runs", type=int, default=500)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)

    trades_path = ROOT / args.trades
    if not trades_path.exists():
        print(f"[error] trades CSV not found: {trades_path}", file=sys.stderr)
        return 2
    out_dir = ROOT / (args.output_dir or trades_path.parent)
    payload = run_prop_replay(
        trades_path,
        account_name=args.account,
        n_runs=args.n_runs,
        instrument=args.instrument,
        output_dir=out_dir,
    )

    print()
    print("=== PROP REPLAY SUMMARY ===")
    print(f"  account:                   {payload['account']}")
    print(f"  instrument:                {payload['instrument']}")
    print(f"  n_trades:                  {payload['n_trades']}")
    chrono = payload["chronological_replay"]
    print(f"  chronological outcome:     {chrono.get('outcome')}")
    print(f"  chronological breach:      {chrono.get('breach')}")
    print(f"  chronological end_balance: {chrono.get('end_balance')}")
    mc = payload["monte_carlo"]
    print(f"  MC pass_probability:       {mc['pass_probability']:.3f}  "
          f"CI {tuple(round(x, 3) for x in mc['pass_probability_ci'])}")
    print(f"  MC blowup_probability:     {mc['blowup_probability']:.3f}  "
          f"CI {tuple(round(x, 3) for x in mc['blowup_probability_ci'])}")
    print(f"  MC median_days_to_pass:    {mc.get('median_days_to_pass')}")
    print(f"  MC median_end_balance:     {mc.get('median_end_balance')}")
    payout = payload["payout"]
    print(f"  payout first_payout_prob:  {payout['first_payout_probability']:.3f}")
    cert = payload["cert"]
    print(f"  certified:                 {cert['passes']}")
    if cert["failures"]:
        for fr in cert["failures"]:
            print(f"    - {fr}")
    print(f"  output: {out_dir}/prop_replay.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
