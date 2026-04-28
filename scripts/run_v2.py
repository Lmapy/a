"""v2 runner.

End-to-end:
  1. Validate data.
  2. Enumerate a candidate grid (one spec per entry_model × selectivity preset).
  3. For each spec:
       a. walk-forward (>=20 disjoint folds on long H4)
       b. holdout (matched H4 + M15) with realistic execution
       c. holdout under STRESS execution (slippage x2, spread x1.5)
       d. shuffled-outcome test
       e. random-baseline test
       f. multi-year holdout
       g. prop-firm simulation for 25k/50k/150k
       h. strict certification
  4. Apply Benjamini-Hochberg FDR across all p-values.
  5. Write reports/* CSVs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.trade_metrics import all_metrics
from core.types import Spec
from data.loader import load_all
from data.validator import run_full_validation
from execution.executor import ExecutionModel, run as run_exec
from prop.simulator import simulate_all
from regime.filters import regime_breakdown
from validation.certify import certify
from validation.holdout import yearly_segments
from validation.statistical_tests import (
    benjamini_hochberg, random_baseline_test, shuffled_outcome_test,
)
from validation.walkforward import walk_forward, WFConfig

OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def candidate_grid() -> list[Spec]:
    """Slim grid: one spec per entry_model × selectivity preset.

    The point of v2 isn't a brute-force sweep -- we keep the grid small
    enough that the heavy stats (shuffle/random baseline/yearly/prop)
    can run on every candidate.
    """
    selectivity_presets = [
        # tight (good walk-forward champion from v1)
        {
            "id": "selectivity_tight",
            "filters": [
                {"type": "body_atr", "min": 0.5, "atr_n": 14},
                {"type": "regime", "ma_n": 50, "side": "with"},
            ],
        },
        # medium (no body filter, just regime)
        {
            "id": "selectivity_med",
            "filters": [
                {"type": "regime", "ma_n": 50, "side": "with"},
            ],
        },
        # session-restricted (NY hours)
        {
            "id": "selectivity_ny",
            "filters": [
                {"type": "body_atr", "min": 0.5, "atr_n": 14},
                {"type": "regime", "ma_n": 50, "side": "with"},
                {"type": "session", "hours_utc": [12, 16]},
            ],
        },
    ]
    entries = [
        {"type": "touch_entry"},
        {"type": "reaction_close"},
        {"type": "fib_limit_entry", "level": 0.382},
        {"type": "fib_limit_entry", "level": 0.5},
        {"type": "fib_limit_entry", "level": 0.618},
        {"type": "zone_midpoint_limit"},
        {"type": "minor_structure_break", "lookback": 3},
        {"type": "delayed_entry_1"},
        {"type": "delayed_entry_2"},
        {"type": "sweep_reclaim"},
    ]
    stops = [
        {"type": "prev_h4_open"},
        {"type": "prev_h4_extreme"},
        {"type": "h4_atr", "mult": 1.0, "atr_n": 14},
    ]
    out: list[Spec] = []
    for sp in selectivity_presets:
        for e in entries:
            for s in stops:
                idp = f"{sp['id']}_{e['type']}"
                if e['type'] == 'fib_limit_entry':
                    idp += f"_lvl{e['level']}"
                idp += f"_{s['type']}"
                if s['type'] == 'h4_atr':
                    idp += f"x{s['mult']}"
                out.append(Spec(
                    id=idp,
                    filters=sp["filters"],
                    entry=e,
                    stop=s,
                    exit={"type": "h4_close"},
                ))
    return out


def evaluate_spec(spec: Spec, ds: dict) -> dict:
    h4_long, h4, m15 = ds["h4_long"], ds["h4"], ds["m15"]
    weeks = (h4["time"].iloc[-1] - h4["time"].iloc[0]).total_seconds() / 604_800.0

    # 1. walk-forward
    wf = walk_forward(spec, h4_long, WFConfig(train_months=9, test_months=3,
                                              step_months=3, min_folds=20))

    # 2. holdout normal exec
    trades = run_exec(spec, h4, m15, ExecutionModel())
    ho = all_metrics(trades, window_weeks=weeks)

    # 3. holdout stress exec
    stress_trades = run_exec(spec, h4, m15, ExecutionModel().stress())
    stress = all_metrics(stress_trades, window_weeks=weeks)

    # 4-5. statistical tests on the realistic-exec trades
    sh = shuffled_outcome_test(trades, n_perm=500)
    rb = random_baseline_test(trades, h4_long, n_runs=200)

    # 6. multi-year holdout
    ym = yearly_segments(spec, h4_long, min_positive_years=3)

    # 7. prop firm sim
    prop = simulate_all(trades)

    # 8. certify
    src = "unknown"
    if "dataset_source" in h4.columns and not h4.empty:
        src = str(h4["dataset_source"].iloc[0])

    cr = certify(
        wf=wf,
        holdout_metrics=ho,
        holdout_stress=stress,
        stat_shuffle=sh,
        stat_random=rb,
        yearly=ym,
        dataset_source=src,
    )

    # regime breakdown for reporting
    rb_breakdown = regime_breakdown(trades, h4)

    return {
        "id": spec.id,
        "spec": spec.to_json(),
        "wf": wf,
        "ho": ho,
        "stress": stress,
        "stat_shuffle": sh,
        "stat_random": rb,
        "yearly": ym,
        "prop": prop,
        "cert": {"certified": cr.certified, "failures": cr.failures, "detail": cr.detail},
        "regime_breakdown": rb_breakdown.to_dict(orient="records") if not rb_breakdown.empty else [],
        "trades_per_week": ho.get("trades_per_week", 0.0),
    }


def write_reports(results: list[dict]) -> None:
    # Apply BH-FDR across shuffle p-values for a population-corrected significance flag.
    p_shuffle = [r["stat_shuffle"]["p_value"] for r in results]
    fdr_keep = benjamini_hochberg(p_shuffle, q=0.05)

    rows_lb = []
    rows_stats = []
    rows_exec = []
    rows_prop = []
    rows_entry_cmp = []

    for keep, r in zip(fdr_keep, results):
        cert = r["cert"]
        rows_lb.append({
            "id": r["id"],
            "certified": cert["certified"],
            "fdr_significant": bool(keep),
            "wf_folds": r["wf"]["folds"],
            "wf_median_sharpe": r["wf"]["median_sharpe"],
            "wf_pct_positive_folds": r["wf"]["pct_positive_folds"],
            "ho_trades": r["ho"].get("trades", 0),
            "ho_trades_per_week": r["trades_per_week"],
            "ho_total_return": r["ho"].get("total_return", 0.0),
            "ho_sharpe_ann": r["ho"].get("sharpe_ann", 0.0),
            "ho_profit_factor": r["ho"].get("profit_factor", 0.0),
            "ho_max_drawdown": r["ho"].get("max_drawdown", 0.0),
            "biggest_trade_share": r["ho"].get("biggest_trade_share", 0.0),
            "stress_total_return": r["stress"].get("total_return", 0.0),
            "shuffle_p_value": r["stat_shuffle"]["p_value"],
            "random_p_value": r["stat_random"]["p_value"],
            "yearly_positive": r["yearly"]["n_positive_years"],
            "yearly_total": r["yearly"]["n_years"],
            "regime_consistency": r["yearly"]["regime_consistency_score"],
            "prop_25k_passes": r["prop"]["25k"]["passes_25k"],
            "prop_50k_passes": r["prop"]["50k"]["passes_50k"],
            "prop_150k_passes": r["prop"]["150k"]["passes_150k"],
            "prop_25k_blowup": r["prop"]["25k"]["blowup_probability"],
            "fail_reasons": "; ".join(cert["failures"]) if cert["failures"] else "",
            "spec": json.dumps(r["spec"]),
        })

        rows_stats.append({
            "id": r["id"],
            "shuffle_p_value": r["stat_shuffle"]["p_value"],
            "shuffle_passes": r["stat_shuffle"]["passes"],
            "shuffle_real_sharpe": r["stat_shuffle"].get("real_sharpe"),
            "shuffle_p95_sharpe": r["stat_shuffle"].get("shuffled_p95_sharpe"),
            "random_p_value": r["stat_random"].get("p_value"),
            "random_passes": r["stat_random"].get("passes"),
            "random_real_sharpe": r["stat_random"].get("real_sharpe"),
            "random_p95_sharpe": r["stat_random"].get("random_p95_sharpe"),
            "fdr_significant": bool(keep),
        })

        rows_exec.append({
            "id": r["id"],
            "ho_total_return": r["ho"].get("total_return", 0.0),
            "ho_max_drawdown": r["ho"].get("max_drawdown", 0.0),
            "stress_total_return": r["stress"].get("total_return", 0.0),
            "stress_max_drawdown": r["stress"].get("max_drawdown", 0.0),
            "ho_profit_factor": r["ho"].get("profit_factor", 0.0),
            "stress_profit_factor": r["stress"].get("profit_factor", 0.0),
            "passes_stress": r["stress"].get("total_return", 0.0) > 0,
        })

        rows_prop.append({
            "id": r["id"],
            "25k_blowup": r["prop"]["25k"]["blowup_probability"],
            "25k_survival": r["prop"]["25k"]["prop_survival_score"],
            "25k_p50_balance": r["prop"]["25k"]["end_balance_p50"],
            "25k_passes": r["prop"]["25k"]["passes_25k"],
            "50k_blowup": r["prop"]["50k"]["blowup_probability"],
            "50k_survival": r["prop"]["50k"]["prop_survival_score"],
            "50k_p50_balance": r["prop"]["50k"]["end_balance_p50"],
            "50k_passes": r["prop"]["50k"]["passes_50k"],
            "150k_blowup": r["prop"]["150k"]["blowup_probability"],
            "150k_survival": r["prop"]["150k"]["prop_survival_score"],
            "150k_p50_balance": r["prop"]["150k"]["end_balance_p50"],
            "150k_passes": r["prop"]["150k"]["passes_150k"],
        })

        spec = r["spec"]
        rows_entry_cmp.append({
            "id": r["id"],
            "bias_tf": spec["bias_timeframe"],
            "entry_tf": spec["entry_timeframe"],
            "entry_model": spec["entry"]["type"]
                            + (f"_lvl{spec['entry'].get('level')}"
                               if spec['entry']['type'] == 'fib_limit_entry' else ""),
            "stop": spec["stop"]["type"]
                    + (f"x{spec['stop'].get('mult')}"
                       if spec['stop']['type'] == 'h4_atr' else ""),
            "trades_per_week": r["trades_per_week"],
            "win_rate": r["ho"].get("win_rate", 0.0),
            "profit_factor": r["ho"].get("profit_factor", 0.0),
            "sharpe_ann": r["ho"].get("sharpe_ann", 0.0),
            "max_drawdown": r["ho"].get("max_drawdown", 0.0),
            "mae_mean": r["ho"].get("mae_mean", 0.0),
            "mfe_mean": r["ho"].get("mfe_mean", 0.0),
            "entry_efficiency_mean": r["ho"].get("entry_efficiency_mean", 0.0),
            "near_miss_tp_rate": r["ho"].get("near_miss_tp_rate", 0.0),
            "prop_25k_survival": r["prop"]["25k"]["prop_survival_score"],
            "certified": r["cert"]["certified"],
        })

    pd.DataFrame(rows_lb).sort_values(
        ["certified", "fdr_significant", "wf_median_sharpe", "ho_total_return"],
        ascending=[False, False, False, False],
    ).to_csv(OUT / "v2_leaderboard.csv", index=False)
    pd.DataFrame(rows_stats).to_csv(OUT / "v2_statistical.csv", index=False)
    pd.DataFrame(rows_exec).to_csv(OUT / "v2_execution_robustness.csv", index=False)
    pd.DataFrame(rows_prop).to_csv(OUT / "v2_prop_simulation.csv", index=False)
    pd.DataFrame(rows_entry_cmp).to_csv(OUT / "v2_entry_comparison.csv", index=False)

    cert_rows = [r for r in rows_lb if r["certified"]]
    pd.DataFrame(cert_rows).to_csv(OUT / "v2_certified.csv", index=False)


def main() -> None:
    print("[1/4] loading + validating data ...")
    ds = load_all()
    rep = run_full_validation(**ds)
    if rep.errors:
        print("VALIDATION FAILED:")
        for f in rep.errors:
            print(f"  [{f.code}] {f.message}")
        raise SystemExit(2)
    print(f"  validator: {len(rep.errors)} errors, {len(rep.findings)-len(rep.errors)} warnings")

    grid = candidate_grid()
    print(f"[2/4] evaluating {len(grid)} candidate specs ...")
    results: list[dict] = []
    for i, spec in enumerate(grid, 1):
        try:
            r = evaluate_spec(spec, ds)
        except Exception as ex:  # noqa: BLE001
            print(f"  spec {spec.id} crashed: {ex}")
            continue
        results.append(r)
        if i % 5 == 0 or i == len(grid):
            cert_so_far = sum(1 for r in results if r["cert"]["certified"])
            print(f"  evaluated {i}/{len(grid)}  (certified so far: {cert_so_far})")

    print("[3/4] writing reports ...")
    write_reports(results)

    n_cert = sum(1 for r in results if r["cert"]["certified"])
    print(f"[4/4] done. {n_cert} certified out of {len(results)}.")
    print(f"  outputs: {OUT}/v2_leaderboard.csv, v2_certified.csv, v2_statistical.csv, "
          f"v2_execution_robustness.csv, v2_prop_simulation.csv, v2_entry_comparison.csv")


if __name__ == "__main__":
    main()
