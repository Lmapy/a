"""Parameter-sweep runner for the CBR scalp engine.

Reads a YAML "sweep" config that lists alternative values per knob.
Cartesian-products them into concrete CBRGoldScalpConfig instances,
runs each through the engine, and writes a ranked leaderboard.

The runner is intentionally controlled — every cell of the cartesian
product is evaluated, but the YAML's lists are kept short (a few
values per knob) so a full sweep stays under ~200 combos.
"""
from __future__ import annotations

import copy
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import yaml

from strategies.scalp.config import (
    CBRGoldScalpConfig, EntryConfig, ExpansionConfig, HTFBiasConfig,
    SessionConfig, StopTargetConfig, StructureConfig, TriggerConfig,
)
from strategies.scalp.engine import run_backtest
from strategies.scalp.metrics import compute_metrics


# Map sweep-yaml keys to (sub_section_name, field_name).
_FIELD_MAP: dict[str, tuple[str, str]] = {
    "expansion_lookback_bars":         ("expansion", "expansion_lookback_bars"),
    "min_directional_candle_percent":  ("expansion", "min_directional_candle_percent"),
    "min_net_move_atr_multiple":       ("expansion", "min_net_move_atr_multiple"),
    "trigger_mode":                    ("trigger",   "trigger_mode"),
    "pivot_left":                      ("structure", "pivot_left"),
    "pivot_right":                     ("structure", "pivot_right"),
    "entry_mode":                      ("entry",     "entry_mode"),
    "risk_reward":                     ("stop_target","risk_reward"),
    "bias_mode":                       ("htf_bias",  "bias_mode"),
    "execution_window_start":          ("session",   "execution_window_start"),
    "execution_window_end":            ("session",   "execution_window_end"),
}


def expand_sweep(yaml_path: Path | str,
                  base_cfg: CBRGoldScalpConfig
                  ) -> list[tuple[dict, CBRGoldScalpConfig]]:
    """Return [(grid_cell, cfg), ...] from a sweep YAML."""
    text = Path(yaml_path).read_text(encoding="utf-8")
    spec = yaml.safe_load(text) or {}
    grid = spec.get("grid", {})
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    out: list[tuple[dict, CBRGoldScalpConfig]] = []
    for combo in itertools.product(*values):
        cell = dict(zip(keys, combo))
        cfg = _apply_cell(base_cfg, cell)
        out.append((cell, cfg))
    return out


def _apply_cell(base_cfg: CBRGoldScalpConfig, cell: dict
                  ) -> CBRGoldScalpConfig:
    """Return a fresh CBRGoldScalpConfig with `cell` overrides applied."""
    cfg = CBRGoldScalpConfig.from_json(json.loads(base_cfg.to_json_str()))
    for k, v in cell.items():
        if k not in _FIELD_MAP:
            raise ValueError(f"sweep key {k!r} not in _FIELD_MAP")
        section, field = _FIELD_MAP[k]
        sub = getattr(cfg, section)
        setattr(sub, field, v)
    return cfg


def stability_score(metrics: dict, *,
                     min_trades: int = 30,
                     biggest_share_cap: float = 0.25) -> float:
    """0..100 score that penalises overfit-shaped results.

    Rewards:
        positive expectancy
        enough trades (>= min_trades)
        not depending on one outlier (biggest_share <= cap)
        decent profit factor
    Penalises:
        very high drawdown vs total R
    """
    if metrics.get("total_trades", 0) < min_trades:
        return 0.0
    biggest = metrics.get("biggest_trade_share", 1.0)
    if biggest > biggest_share_cap:
        return 0.0
    expectancy = metrics.get("expectancy_r", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    if pf == "inf":
        pf = 5.0
    total_r = metrics.get("total_r", 0.0)
    max_dd = abs(metrics.get("max_drawdown_r", 0.0))
    if expectancy <= 0 or total_r <= 0:
        return 0.0
    score = 0.0
    score += min(1.0, expectancy / 0.5) * 35
    score += min(1.0, pf / 2.0) * 25
    score += min(1.0, total_r / max(max_dd, 1.0)) * 25
    score += min(1.0, metrics.get("total_trades", 0) / 100.0) * 15
    return round(score, 2)


def run_sweep(*, base_yaml: Path | str,
                sweep_yaml: Path | str,
                m1: pd.DataFrame, h1: pd.DataFrame,
                dxy: pd.DataFrame | None,
                output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = CBRGoldScalpConfig.from_yaml(base_yaml)
    cells = expand_sweep(sweep_yaml, base)

    rows = []
    t0 = time.time()
    for i, (cell, cfg) in enumerate(cells, 1):
        results = run_backtest(cfg, m1, h1, dxy=dxy)
        metrics = compute_metrics(results["trades"], results["setups"])
        score = stability_score(metrics)
        # flags
        flags = []
        if metrics.get("total_trades", 0) < 30:
            flags.append("LOW_TRADES")
        if metrics.get("biggest_trade_share", 1.0) > 0.25:
            flags.append("OUTLIER_DEPENDENT")
        if metrics.get("profit_factor", 0.0) == "inf":
            flags.append("PF_INF_LOW_N")
        max_dd = abs(metrics.get("max_drawdown_r", 0.0))
        if max_dd > metrics.get("total_r", 0.0):
            flags.append("HIGH_DD_VS_TOTAL_R")
        rows.append({
            **cell,
            "total_trades": metrics.get("total_trades", 0),
            "win_rate": metrics.get("win_rate", 0.0),
            "expectancy_r": metrics.get("expectancy_r", 0.0),
            "total_r": metrics.get("total_r", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "max_drawdown_r": metrics.get("max_drawdown_r", 0.0),
            "biggest_trade_share": metrics.get("biggest_trade_share", 0.0),
            "stability_score": score,
            "flags": ";".join(flags),
        })
        if i % 5 == 0 or i == len(cells):
            print(f"  [{i}/{len(cells)}] elapsed {(time.time()-t0)/60:.1f}min")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "sweep_leaderboard.csv", index=False)
    # ranked dumps
    if not df.empty:
        df.sort_values("expectancy_r", ascending=False).head(20).to_csv(
            output_dir / "rank_by_expectancy.csv", index=False)
        df.sort_values("profit_factor", ascending=False).head(20).to_csv(
            output_dir / "rank_by_profit_factor.csv", index=False)
        df.sort_values("total_r", ascending=False).head(20).to_csv(
            output_dir / "rank_by_total_r.csv", index=False)
        df.sort_values("max_drawdown_r", ascending=True).head(20).to_csv(
            output_dir / "rank_by_lowest_drawdown.csv", index=False)
        df.sort_values("stability_score", ascending=False).head(20).to_csv(
            output_dir / "rank_by_stability.csv", index=False)
    summary = {
        "n_combos": len(cells),
        "runtime_s": round(time.time() - t0, 1),
        "best_stability": float(df["stability_score"].max()) if not df.empty else 0.0,
        "best_expectancy": float(df["expectancy_r"].max()) if not df.empty else 0.0,
        "n_with_30_plus_trades": int((df["total_trades"] >= 30).sum())
                                   if not df.empty else 0,
    }
    (output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary
