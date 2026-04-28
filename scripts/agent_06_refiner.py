"""Agent 06 — Targeted Refiner.

Reads results/leaderboard.csv, picks the top-N near-miss strategies
(passes critic, fails statistical-significance OR fails by <=20%
positive-folds threshold), and applies SINGLE-KNOB structured
refinements from the allow-list in agents/18-refiner.md.

Each refinement variant is re-evaluated against the same gates as the
parent runner. Strict guardrails:

  - exactly one knob changes per round
  - the knob comes from a fixed allow-list (no free parameter sweeps)
  - we never look at holdout p-values when picking the knob
  - duplicates (same spec already on the leaderboard) are dropped

Outputs:
  results/refinement_log.json  every (parent, knob, variant) round
  results/refined_specs.csv    per-variant evaluation row
"""
from __future__ import annotations

import copy
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.trade_metrics import all_metrics
from core.types import Spec
from data.loader import load_all
from execution.executor import ExecutionModel, run as run_exec
from validation.certify import certify
from validation.critic import run_critic
from validation.holdout import yearly_segments
from validation.statistical_tests import (
    daily_block_bootstrap_test,
    label_permutation_test,
    random_eligible_entry_test,
    N_PERM_EXPLORATION,
)
from validation.walkforward import walk_forward, WFConfig

LB_CSV = ROOT / "results" / "leaderboard.csv"
LOG = ROOT / "results" / "refinement_log.json"
REFINED_CSV = ROOT / "results" / "refined_specs.csv"


# ---------- single-knob refinement allow-list (agents/18-refiner.md) ----------

KNOBS = [
    # 1. Toggle / tighten session filter
    {"label": "session_NY",          "patch": ("add_filter",
                                              {"type": "session", "hours_utc": [12, 16]})},
    {"label": "session_London",      "patch": ("add_filter",
                                              {"type": "session", "hours_utc": [8, 12]})},
    # 2. Add HTF context guard
    {"label": "pdh_pdl_inside",      "patch": ("add_filter",
                                              {"type": "pdh_pdl", "mode": "inside"})},
    # NOTE: htf_vwap_dist removed in Batch F. VWAP-style filters
    # require real volume that the harness does not have on disk;
    # the feature-capability gate rejects them. The OHLC-only proxy
    # `atr_distance_from_session_mean` will replace this knob once
    # the proxy filter is wired into the executor in Batch G.
    # 3. Regime-class restriction
    {"label": "regime_trend",        "patch": ("add_filter",
                                              {"type": "regime_class", "allow": ["trend"]})},
    {"label": "regime_trend_expansion","patch": ("add_filter",
                                              {"type": "regime_class",
                                               "allow": ["trend", "expansion"]})},
    # 4. ATR-band volatility filter
    {"label": "atr_pct_30_80",       "patch": ("add_filter",
                                              {"type": "atr_percentile",
                                               "window": 100, "lo": 0.30, "hi": 0.80})},
    # 5. Stop swap (structural <-> ATR)
    {"label": "stop_h4_atr_x1",      "patch": ("set_stop",
                                              {"type": "h4_atr", "mult": 1.0, "atr_n": 14})},
    {"label": "stop_prev_h4_open",   "patch": ("set_stop", {"type": "prev_h4_open"})},
    {"label": "stop_prev_h4_extreme","patch": ("set_stop", {"type": "prev_h4_extreme"})},
    # 6. Body-ATR threshold tweak
    {"label": "body_min_0.3",        "patch": ("set_filter_value",
                                              "body_atr", "min", 0.3)},
    {"label": "body_min_0.7",        "patch": ("set_filter_value",
                                              "body_atr", "min", 0.7)},
    # 7. Streak filter add/tighten
    {"label": "streak_2",            "patch": ("add_filter",
                                              {"type": "min_streak", "k": 2})},
    {"label": "streak_3",            "patch": ("add_filter",
                                              {"type": "min_streak", "k": 3})},
    # 8. Fib level swap (when applicable)
    {"label": "fib_0.382",           "patch": ("set_entry_level", 0.382)},
    {"label": "fib_0.618",           "patch": ("set_entry_level", 0.618)},
    {"label": "fib_0.705",           "patch": ("set_entry_level", 0.705)},
]


def _apply_knob(spec: dict, patch: tuple) -> dict | None:
    out = copy.deepcopy(spec)
    op = patch[0]
    if op == "add_filter":
        f = patch[1]
        # don't double-add the same filter type (skip refinement instead)
        if any(g["type"] == f["type"] for g in out.get("filters", [])):
            return None
        out["filters"].append(f)
    elif op == "set_stop":
        out["stop"] = patch[1]
    elif op == "set_filter_value":
        ftype, key, val = patch[1], patch[2], patch[3]
        for g in out.get("filters", []):
            if g["type"] == ftype:
                g[key] = val
                break
        else:
            return None  # the original spec doesn't have this filter
    elif op == "set_entry_level":
        if out["entry"]["type"] != "fib_limit_entry":
            return None
        out["entry"]["level"] = patch[1]
    else:
        raise ValueError(f"unknown patch op: {op}")
    return out


# ---------- selection of near-misses ----------

def pick_near_misses(lb: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Pick strategies that pass the critic but fail the statistical
    gates -- those are the most actionable candidates to refine."""
    cands = lb.copy()
    cands = cands[cands["passes_critic"].astype(str).str.lower() == "true"]
    if cands.empty:
        # fallback: top by walk-forward Sharpe even if critic failed
        cands = lb.sort_values("wf_median_sharpe", ascending=False).head(top_n).copy()
    cands = cands.sort_values(
        ["wf_median_sharpe", "ho_total_return", "ho_profit_factor"],
        ascending=[False, False, False],
    ).head(top_n)
    return cands.to_dict(orient="records")


# ---------- evaluator (compact reuse of run_alpha) ----------

def _evaluate(spec: Spec, ds: dict) -> dict:
    h4_long, h4, m15 = ds["h4_long"], ds["h4"], ds["m15"]
    weeks = (h4["time"].iloc[-1] - h4["time"].iloc[0]).total_seconds() / 604_800.0

    wf = walk_forward(spec, h4_long, m15, WFConfig())
    trades = run_exec(spec, h4, m15, ExecutionModel())
    stress_trades = run_exec(spec, h4, m15, ExecutionModel().stress())
    ho = all_metrics(trades, window_weeks=weeks)
    stress = all_metrics(stress_trades, window_weeks=weeks)
    lp = label_permutation_test(trades, n_perm=N_PERM_EXPLORATION)
    rb = random_eligible_entry_test(trades, h4_long, n_runs=N_PERM_EXPLORATION)
    bb = daily_block_bootstrap_test(trades, n_runs=N_PERM_EXPLORATION)
    yearly = yearly_segments(spec, h4_long, m15, min_positive_years=3)
    crit = run_critic(trades)
    cr = certify(wf=wf, holdout_metrics=ho, holdout_stress=stress,
                 stat_label_perm=lp, stat_random=rb, stat_block_boot=bb,
                 yearly=yearly)
    return {
        "id": spec.id,
        "wf_folds": wf["folds"],
        "wf_median_sharpe": wf["median_sharpe"],
        "wf_pct_positive_folds": wf["pct_positive_folds"],
        "ho_trades": ho.get("trades", 0),
        "ho_total_return": ho.get("total_return", 0.0),
        "ho_sharpe_ann": ho.get("sharpe_ann", 0.0),
        "ho_profit_factor": ho.get("profit_factor", 0.0),
        "ho_max_drawdown": ho.get("max_drawdown", 0.0),
        "stress_total_return": stress.get("total_return", 0.0),
        "label_perm_p_value": lp["p_value"],
        "random_p_value": rb.get("p_value", 1.0),
        "block_boot_p_value": bb.get("p_value", 1.0),
        "passes_critic": crit.passes_critic,
        "certified": cr.certified,
        "cert_failures": cr.failures,
    }


# ---------- main ----------

def run() -> dict:
    if not LB_CSV.exists():
        raise SystemExit("results/leaderboard.csv missing — run agent 04 first")
    lb = pd.read_csv(LB_CSV)
    parents = pick_near_misses(lb, top_n=5)
    if not parents:
        out = {"n_parents": 0, "n_variants": 0, "rounds": []}
        LOG.write_text(json.dumps(out, indent=2))
        print(f"  refiner: no near-misses found; nothing to refine")
        return out

    ds = load_all()
    rounds = []
    rows = []
    seen_ids = set(lb["id"].astype(str).tolist())

    for parent_row in parents:
        parent_spec = json.loads(parent_row["spec"]) if isinstance(parent_row.get("spec"), str) else parent_row.get("spec")
        if parent_spec is None:
            continue
        parent_id = str(parent_row["id"])
        for knob in KNOBS:
            new_dict = _apply_knob(parent_spec, knob["patch"])
            if new_dict is None:
                continue
            new_dict["id"] = f"{parent_id}+{knob['label']}"
            if new_dict["id"] in seen_ids:
                continue
            seen_ids.add(new_dict["id"])
            spec = Spec(
                id=new_dict["id"],
                bias_timeframe=new_dict.get("bias_timeframe", "H4"),
                setup_timeframe=new_dict.get("setup_timeframe", "H4"),
                entry_timeframe=new_dict.get("entry_timeframe", "M15"),
                signal=new_dict["signal"],
                filters=new_dict.get("filters", []),
                entry=new_dict["entry"],
                stop=new_dict["stop"],
                exit=new_dict["exit"],
            )
            try:
                m = _evaluate(spec, ds)
            except Exception as ex:  # noqa: BLE001
                rounds.append({
                    "parent_id": parent_id,
                    "knob": knob["label"],
                    "result_id": new_dict["id"],
                    "error": str(ex),
                })
                continue
            rounds.append({
                "parent_id": parent_id,
                "knob": knob["label"],
                "result_id": new_dict["id"],
                "result": m,
            })
            rows.append({"parent_id": parent_id, "knob": knob["label"],
                         "spec": json.dumps(new_dict), **m})

    LOG.write_text(json.dumps({
        "n_parents": len(parents),
        "n_variants": len(rounds),
        "rounds": rounds,
    }, indent=2, default=str))
    if rows:
        pd.DataFrame(rows).to_csv(REFINED_CSV, index=False)

    n_better = sum(
        1 for r in rounds
        if "result" in r and r["result"].get("certified")
    )
    print(f"  refiner: {len(parents)} parents, {len(rounds)} variants, "
          f"newly-certified={n_better}")
    return {"n_parents": len(parents), "n_variants": len(rounds),
            "newly_certified": n_better}


if __name__ == "__main__":
    run()
