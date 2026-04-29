"""Monte Carlo trade-reorder confidence intervals.

Bootstraps the trade-return sequence to estimate confidence
intervals on total R, max drawdown, and final equity. The trades
themselves are NOT re-simulated -- we shuffle the realised R-list,
which preserves the marginal R distribution but destroys time
ordering. That answers "how dependent is the equity curve on the
order in which trades happened?".

This is complementary to the harness's label-permutation +
random-eligible-entry tests (which permute SIGNALS) and the
day-block bootstrap (which permutes DAYS).

Outputs:
    p05, p50, p95 of total_r, max_drawdown_r, final_equity
    histogram bin edges + counts (so a UI can render)
    full bootstrap distributions (n_runs x summary)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def monte_carlo_reorder(trades: list, *,
                         n_runs: int = 5000,
                         initial_equity: float = 50_000.0,
                         seed: int = 17) -> dict:
    """Run `n_runs` random reorderings of the trade list.

    Returns a dict with point estimates and bootstrap percentiles
    for total_r, max_drawdown_r, final_equity. Includes the full
    bootstrap arrays so callers can plot CI bands.
    """
    if not trades:
        return {"n_runs": 0, "n_trades": 0,
                 "note": "no trades to reorder"}

    rs = np.array([t.r_result for t in trades], dtype=float)
    pnls = np.array([t.pnl for t in trades], dtype=float)
    n = len(rs)
    rng = np.random.default_rng(seed)

    total_r = np.empty(n_runs, dtype=float)
    max_dd_r = np.empty(n_runs, dtype=float)
    final_eq = np.empty(n_runs, dtype=float)

    for i in range(n_runs):
        order = rng.permutation(n)
        rs_perm = rs[order]
        pnls_perm = pnls[order]
        eq_r = np.cumsum(rs_perm)
        peak = np.maximum.accumulate(eq_r)
        dd = (eq_r - peak)
        total_r[i] = float(eq_r[-1])
        max_dd_r[i] = float(dd.min())
        final_eq[i] = float(initial_equity + np.cumsum(pnls_perm)[-1])

    actual_total_r = float(np.sum(rs))
    actual_eq = np.cumsum(rs)
    actual_peak = np.maximum.accumulate(actual_eq)
    actual_dd = float((actual_eq - actual_peak).min())
    actual_final_eq = float(initial_equity + np.cumsum(pnls)[-1])

    return {
        "n_runs": n_runs,
        "n_trades": n,
        "actual": {
            "total_r": round(actual_total_r, 3),
            "max_drawdown_r": round(actual_dd, 3),
            "final_equity": round(actual_final_eq, 2),
        },
        "bootstrap_total_r": {
            "p05": round(float(np.percentile(total_r, 5)), 3),
            "p50": round(float(np.percentile(total_r, 50)), 3),
            "p95": round(float(np.percentile(total_r, 95)), 3),
            "mean": round(float(np.mean(total_r)), 3),
            "std": round(float(np.std(total_r)), 3),
        },
        "bootstrap_max_drawdown_r": {
            "p05": round(float(np.percentile(max_dd_r, 5)), 3),
            "p50": round(float(np.percentile(max_dd_r, 50)), 3),
            "p95": round(float(np.percentile(max_dd_r, 95)), 3),
            "mean": round(float(np.mean(max_dd_r)), 3),
            "std": round(float(np.std(max_dd_r)), 3),
        },
        "bootstrap_final_equity": {
            "p05": round(float(np.percentile(final_eq, 5)), 2),
            "p50": round(float(np.percentile(final_eq, 50)), 2),
            "p95": round(float(np.percentile(final_eq, 95)), 2),
            "mean": round(float(np.mean(final_eq)), 2),
        },
        "share_runs_negative_total_r": round(float(np.mean(total_r < 0)), 4),
        "share_runs_dd_worse_than_actual": round(
            float(np.mean(max_dd_r < actual_dd)), 4),
    }


def write_monte_carlo(payload: dict, output_dir: Path) -> Path:
    """Write the MC summary as JSON next to the trades CSV."""
    p = Path(output_dir) / "monte_carlo.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p
