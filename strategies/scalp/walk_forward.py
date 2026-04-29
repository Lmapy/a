"""Walk-forward / out-of-sample split for the CBR scalp engine.

Default split: first 70% of bars = in-sample, last 30% = out-of-sample.
Both halves run through the same engine; metrics are reported for each
plus a degradation ratio.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from strategies.scalp.config import CBRGoldScalpConfig
from strategies.scalp.engine import run_backtest
from strategies.scalp.metrics import compute_metrics


def run_walk_forward(*, cfg: CBRGoldScalpConfig,
                      m1: pd.DataFrame, h1: pd.DataFrame,
                      dxy: pd.DataFrame | None = None,
                      train_fraction: float = 0.70,
                      output_dir: Path | None = None) -> dict:
    """Split the M1 frame chronologically and run the engine on each
    half. Returns a dict with `in_sample`, `out_of_sample`, and
    `degradation` blocks."""
    if not 0.1 <= train_fraction <= 0.95:
        raise ValueError("train_fraction must be in [0.1, 0.95]")
    n = len(m1)
    if n < 1000:
        raise ValueError(f"m1 too short for walk-forward: {n} bars")
    cut_idx = int(n * train_fraction)
    cut_ts = pd.Timestamp(m1["time"].iloc[cut_idx])
    in_sample_m1 = m1.iloc[:cut_idx].reset_index(drop=True)
    out_sample_m1 = m1.iloc[cut_idx:].reset_index(drop=True)
    # mirror the cut on h1
    in_sample_h1 = h1[h1["time"] <= cut_ts].reset_index(drop=True)
    out_sample_h1 = h1[h1["time"] >= cut_ts - pd.Timedelta(hours=2)
                        ].reset_index(drop=True)

    is_results = run_backtest(cfg, in_sample_m1, in_sample_h1, dxy=dxy)
    oos_results = run_backtest(cfg, out_sample_m1, out_sample_h1, dxy=dxy)
    is_metrics = compute_metrics(is_results["trades"], is_results["setups"])
    oos_metrics = compute_metrics(oos_results["trades"], oos_results["setups"])

    degradation = _degradation(is_metrics, oos_metrics)
    payload = {
        "split_at": cut_ts.isoformat(),
        "train_fraction": train_fraction,
        "in_sample": {
            "n_bars": len(in_sample_m1),
            "metrics": is_metrics,
            "runtime_s": is_results["runtime_s"],
        },
        "out_of_sample": {
            "n_bars": len(out_sample_m1),
            "metrics": oos_metrics,
            "runtime_s": oos_results["runtime_s"],
        },
        "degradation": degradation,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "walkforward_summary.json").write_text(
            json.dumps(payload, indent=2, default=str))
        # save the trade ledgers for inspection
        pd.DataFrame([t.to_dict() for t in is_results["trades"]]).to_csv(
            output_dir / "walkforward_in_sample_trades.csv", index=False)
        pd.DataFrame([t.to_dict() for t in oos_results["trades"]]).to_csv(
            output_dir / "walkforward_out_sample_trades.csv", index=False)
    return payload


def _degradation(is_m: dict, oos_m: dict) -> dict:
    """Compare IS vs OOS metrics; flag if OOS is materially worse."""
    is_t = is_m.get("total_trades", 0)
    oos_t = oos_m.get("total_trades", 0)
    is_e = is_m.get("expectancy_r", 0.0)
    oos_e = oos_m.get("expectancy_r", 0.0)
    is_pf = is_m.get("profit_factor", 0.0)
    oos_pf = oos_m.get("profit_factor", 0.0)

    e_ratio = (oos_e / is_e) if is_e not in (0, "inf") else None
    pf_ratio = (oos_pf / is_pf) if is_pf not in (0, "inf") else None

    is_dd = abs(is_m.get("max_drawdown_r", 0.0))
    oos_dd = abs(oos_m.get("max_drawdown_r", 0.0))
    dd_ratio = (oos_dd / is_dd) if is_dd > 0 else None

    flags = []
    if oos_e <= 0 and is_e > 0:
        flags.append("OOS_NEGATIVE_EXPECTANCY")
    if e_ratio is not None and e_ratio < 0.5 and is_e > 0:
        flags.append("EXPECTANCY_DEGRADED_50PLUS_PCT")
    if pf_ratio is not None and pf_ratio < 0.6:
        flags.append("PROFIT_FACTOR_DEGRADED")
    if dd_ratio is not None and dd_ratio > 1.5:
        flags.append("DRAWDOWN_WORSE_THAN_IN_SAMPLE")
    if is_t > 0 and oos_t == 0:
        flags.append("OOS_HAS_NO_TRADES")
    return {
        "in_sample_trades": is_t,
        "out_of_sample_trades": oos_t,
        "expectancy_in_sample": is_e,
        "expectancy_out_of_sample": oos_e,
        "expectancy_ratio": round(e_ratio, 3) if e_ratio is not None else None,
        "profit_factor_in_sample": is_pf,
        "profit_factor_out_of_sample": oos_pf,
        "profit_factor_ratio": round(pf_ratio, 3) if pf_ratio is not None else None,
        "drawdown_ratio": round(dd_ratio, 3) if dd_ratio is not None else None,
        "flags": flags,
        "stable": not flags,
    }
