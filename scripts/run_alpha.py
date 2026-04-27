"""Agentic alpha-discovery pipeline runner.

Pipeline (each step writes a structured JSON output and is independently
auditable):

  01 data_auditor        validate sources + manifest + alignment
  02 hypothesis_generator emit hypotheses.json
  03 spec_builder        emit generated_specs.json
  04 backtest_runner     run each runnable spec through the realistic
                         executor + walk-forward + holdout
  05 robustness_critic   top-trade removal / worst-week / session-split
                         / consecutive losses / removed-best-year
  06 refiner             (markdown spec only — see agents/18-refiner.md)
  07 certifier           strict gate -> certified_alpha_strategies.json
  08 portfolio_builder   correlation-cluster certified strategies
  09 reporter            update README + audit PDF

This is the end-to-end equivalent of `make report`.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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
from validation.critic import run_critic
from validation.holdout import yearly_segments
from validation.statistical_tests import (
    benjamini_hochberg, random_baseline_test, shuffled_outcome_test,
)
from validation.walkforward import walk_forward, WFConfig

# Sub-runners (agents 02 + 03 + 08).
from scripts import agent_02_hypothesis, agent_03_spec_builder, agent_08_portfolio  # type: ignore

OUT = ROOT / "results"
ALPHA_TRADES = OUT / "alpha_trades"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA_TRADES.mkdir(parents=True, exist_ok=True)

LEADERBOARD_CSV = OUT / "leaderboard.csv"
CERTIFIED_JSON = OUT / "certified_alpha_strategies.json"
CRITIC_JSON = OUT / "critic_report.json"
DATA_AUDIT_JSON = OUT / "agent_data_audit.json"
PROP_JSON = OUT / "prop_simulation.json"


# ---------- agent 01 ----------

def agent_01_data_audit(ds: dict) -> dict:
    rep = run_full_validation(**ds)
    manifest_path = OUT / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    out = {
        "generated_by": "agent_01_data_auditor",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validator_findings": rep.to_rows(),
        "n_errors": len(rep.errors),
        "n_warnings": len([f for f in rep.findings if f.severity == "warning"]),
        "manifest_present": manifest_path.exists(),
        "manifest": manifest,
    }
    DATA_AUDIT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"  agent 01 -> {DATA_AUDIT_JSON}  errors={out['n_errors']}  "
          f"warnings={out['n_warnings']}")
    return out


# ---------- agent 04 + 05 + 07 (per-spec) ----------

def evaluate_one(spec_obj: dict, ds: dict) -> dict:
    """Run one runnable spec through executor + walk-forward + critic + certifier."""
    spec_dict = spec_obj["spec"]
    spec = Spec(
        id=spec_dict["id"],
        bias_timeframe=spec_dict.get("bias_timeframe", "H4"),
        setup_timeframe=spec_dict.get("setup_timeframe", "H4"),
        entry_timeframe=spec_dict.get("entry_timeframe", "M15"),
        signal=spec_dict["signal"],
        filters=spec_dict.get("filters", []),
        entry=spec_dict["entry"],
        stop=spec_dict["stop"],
        exit=spec_dict["exit"],
        risk=spec_dict.get("risk", {"per_trade_pct": 0.5}),
        cost_bps=spec_dict.get("cost_bps", 1.5),
    )

    h4_long, h4, m15 = ds["h4_long"], ds["h4"], ds["m15"]
    weeks = (h4["time"].iloc[-1] - h4["time"].iloc[0]).total_seconds() / 604_800.0

    wf = walk_forward(spec, h4_long, WFConfig())
    trades = run_exec(spec, h4, m15, ExecutionModel())
    stress_trades = run_exec(spec, h4, m15, ExecutionModel().stress())
    ho = all_metrics(trades, window_weeks=weeks)
    stress = all_metrics(stress_trades, window_weeks=weeks)

    sh = shuffled_outcome_test(trades, n_perm=300)
    rb = random_baseline_test(trades, h4_long, n_runs=200)
    yearly = yearly_segments(spec, h4_long, min_positive_years=3)
    prop = simulate_all(trades)

    critic = run_critic(trades)
    cr = certify(
        wf=wf, holdout_metrics=ho, holdout_stress=stress,
        stat_shuffle=sh, stat_random=rb, yearly=yearly,
    )
    rb_break = regime_breakdown(trades, h4)

    # Save per-strategy trade ledger so agent 08 can load it.
    if trades:
        tdf = pd.DataFrame([{
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "dir": t.direction, "entry": t.entry, "exit": t.exit,
            "pnl": t.pnl, "ret": t.ret, "fill_kind": t.fill_kind,
        } for t in trades])
        (ALPHA_TRADES / f"{spec.id}.csv").write_text(tdf.to_csv(index=False))

    return {
        "id": spec.id,
        "family_id": spec_obj["family_id"],
        "spec": spec_dict,
        "wf": wf,
        "ho": ho,
        "stress": stress,
        "stat_shuffle": sh,
        "stat_random": rb,
        "yearly": yearly,
        "prop": prop,
        "critic": {"passes_critic": critic.passes_critic,
                   "failure_reasons": critic.failure_reasons,
                   "detail": critic.detail},
        "cert": {"certified": cr.certified, "failures": cr.failures, "detail": cr.detail},
        "regime_breakdown": rb_break.to_dict(orient="records") if not rb_break.empty else [],
        "trades_per_week": ho.get("trades_per_week", 0.0),
    }


# ---------- main ----------

def main() -> None:
    print("[agent 01] data audit ...")
    ds = load_all()
    audit = agent_01_data_audit(ds)
    if audit["n_errors"] > 0:
        raise SystemExit(2)

    print("[agent 02] generating hypotheses ...")
    agent_02_hypothesis.run()

    print("[agent 03] building specs from hypotheses ...")
    agent_03_spec_builder.run()
    specs_doc = json.loads((OUT / "generated_specs.json").read_text())
    runnable = [s for s in specs_doc["specs"] if s["status"] == "runnable"]
    print(f"  runnable={len(runnable)}  skipped={specs_doc['n_skipped']}")

    print("[agent 04+05+07] backtest + critic + certify per spec ...")
    results: list[dict] = []
    for i, spec_obj in enumerate(runnable, 1):
        try:
            results.append(evaluate_one(spec_obj, ds))
        except Exception as ex:  # noqa: BLE001
            print(f"  spec {spec_obj['spec']['id']} crashed: {ex}")
            continue
        if i % 5 == 0 or i == len(runnable):
            n_cert = sum(1 for r in results
                         if r["cert"]["certified"] and r["critic"]["passes_critic"])
            print(f"  evaluated {i}/{len(runnable)}  (passing both gates: {n_cert})")

    # FDR across the realistic-exec shuffled-outcome p-values.
    p_shuffle = [r["stat_shuffle"]["p_value"] for r in results]
    fdr_keep = benjamini_hochberg(p_shuffle, q=0.05) if p_shuffle else []

    # ---------- write reports ----------
    rows_lb = []
    rows_critic = []
    rows_prop = []
    certified_full: list[dict] = []
    for keep, r in zip(fdr_keep, results):
        passes_both = r["cert"]["certified"] and r["critic"]["passes_critic"]
        row = {
            "id": r["id"],
            "family_id": r["family_id"],
            "certified": r["cert"]["certified"],
            "passes_critic": r["critic"]["passes_critic"],
            "fdr_significant": bool(keep),
            "passes_all_gates": passes_both and bool(keep),
            "wf_folds": r["wf"]["folds"],
            "wf_median_sharpe": r["wf"]["median_sharpe"],
            "wf_pct_positive_folds": r["wf"]["pct_positive_folds"],
            "ho_trades": r["ho"].get("trades", 0),
            "ho_trades_per_week": r["trades_per_week"],
            "ho_total_return": r["ho"].get("total_return", 0.0),
            "ho_sharpe_ann": r["ho"].get("sharpe_ann", 0.0),
            "ho_profit_factor": r["ho"].get("profit_factor", 0.0),
            "ho_max_drawdown": r["ho"].get("max_drawdown", 0.0),
            "stress_total_return": r["stress"].get("total_return", 0.0),
            "shuffle_p_value": r["stat_shuffle"]["p_value"],
            "random_p_value": r["stat_random"]["p_value"],
            "yearly_positive": r["yearly"]["n_positive_years"],
            "yearly_total": r["yearly"]["n_years"],
            "prop_25k_passes": r["prop"]["25k"]["passes_25k"],
            "cert_failures": "; ".join(r["cert"]["failures"]) if r["cert"]["failures"] else "",
            "critic_failures": "; ".join(r["critic"]["failure_reasons"])
                                if r["critic"]["failure_reasons"] else "",
            "spec": json.dumps(r["spec"]),
        }
        rows_lb.append(row)
        rows_critic.append({
            "strategy_id": r["id"],
            "passes_critic": r["critic"]["passes_critic"],
            "failure_reasons": r["critic"]["failure_reasons"],
            "detail": r["critic"]["detail"],
        })
        rows_prop.append({
            "strategy_id": r["id"],
            **{k: v for k, v in r["prop"].items()},
        })
        if passes_both:
            certified_full.append({**row, "spec": r["spec"]})

    # leaderboard.csv
    df_lb = pd.DataFrame(rows_lb)
    if not df_lb.empty:
        df_lb = df_lb.sort_values(
            ["passes_all_gates", "wf_median_sharpe", "ho_total_return"],
            ascending=[False, False, False],
        )
    df_lb.to_csv(LEADERBOARD_CSV, index=False)

    CRITIC_JSON.write_text(json.dumps({
        "generated_by": "agent_05_robustness_critic",
        "n": len(rows_critic),
        "critic_results": rows_critic,
    }, indent=2, default=str))

    PROP_JSON.write_text(json.dumps({
        "generated_by": "agent_07b_prop_simulator",
        "n": len(rows_prop),
        "results": rows_prop,
    }, indent=2, default=str))

    CERTIFIED_JSON.write_text(json.dumps({
        "generated_by": "agent_07_certifier",
        "policy": "Strategy must pass strict cert (>=20 wf folds, "
                  "median Sharpe >0.5, >=60% positive folds, PF >1.2, "
                  "DD>= -0.20, biggest_trade_share <=0.20, shuffle p<0.05, "
                  "random_baseline p<0.05, profitable under stress, "
                  ">=3 positive yearly slices) AND robustness critic "
                  "(top-1 removal, top-5%-removal, worst-week, "
                  "consecutive-loss, session-split, removed-best-year) "
                  "AND BH-FDR significance.",
        "n_certified": len(certified_full),
        "certified": certified_full,
    }, indent=2, default=str))

    print(f"  agent 07 -> {CERTIFIED_JSON}  certified={len(certified_full)}")

    print("[agent 06] targeted refiner ...")
    try:
        from scripts import agent_06_refiner  # type: ignore
        agent_06_refiner.run()
    except Exception as ex:  # noqa: BLE001
        print(f"  refiner skipped: {ex}")

    print("[agent 08] portfolio clustering ...")
    agent_08_portfolio.run()

    print("[alpha judge] meta-analysis ...")
    try:
        from scripts import agent_alpha_judge  # type: ignore
        agent_alpha_judge.run()
    except Exception as ex:  # noqa: BLE001
        print(f"  alpha_judge skipped: {ex}")

    print(f"\n=== alpha pipeline complete ===")
    print(f"  evaluated:     {len(results)}")
    print(f"  fdr_sig:       {sum(rows_lb[i]['fdr_significant'] for i in range(len(rows_lb)))}")
    print(f"  passes critic: {sum(1 for r in rows_lb if r['passes_critic'])}")
    print(f"  certified:     {sum(1 for r in rows_lb if r['certified'])}")
    print(f"  pass ALL:      {sum(1 for r in rows_lb if r['passes_all_gates'])}")
    print(f"  outputs:       {LEADERBOARD_CSV}, {CERTIFIED_JSON}, {CRITIC_JSON}, "
          f"{PROP_JSON}, {OUT}/alpha_portfolio.json, {OUT}/agent_data_audit.json")


if __name__ == "__main__":
    main()
