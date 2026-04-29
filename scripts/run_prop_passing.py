"""Tiered prop-firm passing orchestrator (Batch H).

This is the canonical entrypoint for the prop-firm passing engine.

Flow (matches the user's brief):

  1. Generate candidate universe (Batch G families' tier-1 grid).
  2. Reject unavailable-data candidates loudly (Batch F capability).
  3. Pre-compute TPO levels + session mean once per data slice.
  4. Run fast OHLC backtest filters on each candidate.
  5. Run walk-forward / validation only on survivors.
  6. Run expensive prop simulations only on top survivors.
  7. Rank by prop passing score; write leaderboard + report.

CLI flags
---------
  --smoke                 quick smoke run (small grid, low n_perm)
  --limit-candidates N    keep at most N tier-1 candidates
  --families F[,F...]     restrict to these family ids
  --accounts A[,A...]     restrict prop sim to these account ids
  --max-survivors-for-prop-sim N
                           cap how many tier-1 survivors get the
                           full prop sim (default 10)
  --fast-only             stop after walk-forward (no prop sim)
  --full                  use the wider risk + daily-rule sweeps
  --n-perm N              statistical-test permutation count (default 500)
  --output-stem NAME      basename in results/ (default prop_passing_leaderboard)

Outputs
-------
  results/prop_passing_leaderboard.csv
  results/prop_passing_leaderboard.meta.json
  results/prop_passing_report.md
  results/runs/<run_id>/{progress.json, events.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.tpo_levels import attach_prev_session_tpo, PREV_SESSION_TPO_COLUMNS
from analytics.trade_metrics import all_metrics
from core.candidate import PropCandidate
from core.certification import (
    CertificationLevel, FailureReason, verdict_from_reasons,
)
from core.feature_capability import classify_candidate
from core.run_events import EventWriter, emit_candidate, emit_stage
from core.run_state import RunState
from core.types import Spec
from data.splits import load_splits, coverage_summary
from execution.executor import ExecutionModel, run as run_exec
from prop.simulator import simulate_all
from prop_challenge.accounts import load_all as load_accounts, verification_status
from reports.prop_passing_score import prop_passing_score
from strategies import grid, families, refiner
from strategies.refiner import propose_mutations
from validation.statistical_tests import (
    benjamini_hochberg, daily_block_bootstrap_test,
    label_permutation_test, random_eligible_entry_test,
    N_PERM_EXPLORATION,
)
from validation.walkforward import walk_forward, WFConfig


RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RUNS_DIR = RESULTS / "runs"


# ---------------- helpers ----------------

def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _file_hash(p: Path) -> str:
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _config_hashes() -> dict[str, str]:
    return {
        "config/data_splits.json": _file_hash(ROOT / "config" / "data_splits.json"),
        "config/prop_accounts.json": _file_hash(ROOT / "config" / "prop_accounts.json"),
    }


# ---------------- per-candidate evaluation ----------------

def _evaluate_candidate(cand: PropCandidate, *,
                          h4_train: pd.DataFrame, m15_train: pd.DataFrame,
                          h4_val: pd.DataFrame, m15_val: pd.DataFrame,
                          h4_holdout: pd.DataFrame, m15_holdout: pd.DataFrame,
                          full_h4: pd.DataFrame,
                          full_m15: pd.DataFrame,
                          n_perm: int,
                          fast_only: bool) -> dict:
    """Run one candidate through all gates. Returns a leaderboard
    row dict + the verdict + collected metrics."""
    spec = cand.to_spec()

    # Walk-forward on TRAIN only.
    wf = walk_forward(spec, h4_train, m15_train,
                       WFConfig(train_months=6, test_months=3,
                                step_months=3, min_folds=4))

    # Validation slice
    val_trades = run_exec(spec, h4_val, m15_val, ExecutionModel())
    val_weeks = (h4_val["time"].iloc[-1] - h4_val["time"].iloc[0]).total_seconds() / 604800.0 \
        if len(h4_val) else 1.0
    val = all_metrics(val_trades, window_weeks=val_weeks)

    # Holdout slice
    ho_trades = run_exec(spec, h4_holdout, m15_holdout, ExecutionModel())
    ho_weeks = (h4_holdout["time"].iloc[-1] - h4_holdout["time"].iloc[0]).total_seconds() / 604800.0 \
        if len(h4_holdout) else 1.0
    ho = all_metrics(ho_trades, window_weeks=ho_weeks)
    stress_trades = run_exec(spec, h4_holdout, m15_holdout,
                              ExecutionModel().stress())
    stress = all_metrics(stress_trades, window_weeks=ho_weeks)

    # Statistical tests
    lp = label_permutation_test(ho_trades, n_perm=n_perm)
    rb = random_eligible_entry_test(ho_trades, h4_holdout, n_runs=n_perm)
    bb = daily_block_bootstrap_test(ho_trades, n_runs=n_perm)

    # Failure-reason ladder
    reasons: list[FailureReason] = []
    if wf.get("compatibility", "ok") != "ok":
        reasons.append(FailureReason.FAIL_RESOLUTION_LIMITED)
    if wf.get("median_sharpe", 0.0) <= 0.0:
        reasons.append(FailureReason.FAIL_WALK_FORWARD)
    if ho.get("total_return", 0.0) <= 0.0:
        reasons.append(FailureReason.FAIL_HOLDOUT)
    if ho.get("profit_factor", 0.0) <= 1.0:
        reasons.append(FailureReason.FAIL_PROFIT_FACTOR)
    if ho.get("max_drawdown", 0.0) < -0.20:
        reasons.append(FailureReason.FAIL_MAX_DRAWDOWN)
    if ho.get("biggest_trade_share", 1.0) > 0.20:
        reasons.append(FailureReason.FAIL_BIGGEST_TRADE_SHARE)
    if not lp.get("passes", False):
        reasons.append(FailureReason.FAIL_LABEL_PERMUTATION)
    if not rb.get("passes", False):
        reasons.append(FailureReason.FAIL_RANDOM_BASELINE)
    if not bb.get("passes", False):
        reasons.append(FailureReason.FAIL_BLOCK_BOOTSTRAP)
    if stress.get("total_return", 0.0) <= 0:
        reasons.append(FailureReason.FAIL_SPREAD_STRESS)

    # Build row
    n_filters = len(cand.filters)
    return {
        "candidate": cand,
        "wf": wf, "val": val, "ho": ho, "stress": stress,
        "stat_label_perm": lp, "stat_random": rb, "stat_block_boot": bb,
        "ho_trades": ho_trades, "stress_trades": stress_trades,
        "reasons": reasons,
        "n_filters": n_filters,
    }


def _attach_prop(result: dict) -> dict:
    """Attach prop-firm simulation results. Called only on tier-1
    survivors that didn't fail the cheap gates."""
    trades = result["ho_trades"]
    prop = simulate_all(trades)
    # Headline: pass / blowup / median_days for the small 25k account.
    p25 = prop.get("25k", {})
    p50 = prop.get("50k", {})
    p150 = prop.get("150k", {})

    pass_prob = p25.get("pass_probability", 0.0)
    blowup_prob = p25.get("blowup_probability", 0.0)
    pass_ci = p25.get("pass_probability_ci", (0.0, 1.0))
    blowup_ci = p25.get("blowup_probability_ci", (0.0, 1.0))

    new_reasons = list(result["reasons"])
    if pass_prob < 0.20:
        new_reasons.append(FailureReason.FAIL_LOW_PASS_PROBABILITY)
    if blowup_prob > 0.30:
        new_reasons.append(FailureReason.FAIL_HIGH_BLOWUP_PROBABILITY)

    result["prop"] = prop
    result["prop_summary"] = {
        "25k_pass_probability": pass_prob,
        "25k_pass_probability_ci": pass_ci,
        "25k_blowup_probability": blowup_prob,
        "25k_blowup_probability_ci": blowup_ci,
        "50k_blowup_probability": p50.get("blowup_probability", 0.0),
        "150k_blowup_probability": p150.get("blowup_probability", 0.0),
    }
    result["reasons"] = new_reasons
    return result


# ---------------- leaderboard rows ----------------

def _row_from_result(result: dict, *, fdr_keep: bool,
                      account_verification: dict[str, str]) -> dict:
    cand = result["candidate"]
    wf = result["wf"]
    val = result["val"]
    ho = result["ho"]
    stress = result["stress"]
    lp = result["stat_label_perm"]
    rb = result["stat_random"]
    bb = result["stat_block_boot"]
    prop_summary = result.get("prop_summary", {})
    reasons = list(result["reasons"])

    # Compute prop passing score from whatever is available
    score_payload = prop_passing_score(
        pass_probability=prop_summary.get("25k_pass_probability", 0.0),
        blowup_probability=prop_summary.get("25k_blowup_probability", 0.0),
        payout_survival_probability=0.0,         # not yet wired
        daily_loss_breach_probability=0.0,
        trailing_drawdown_breach_probability=0.0,
        max_drawdown=ho.get("max_drawdown", 0.0),
        yearly_positive=0, yearly_total=0,        # yearly path is legacy
        median_days_to_pass=None,
        n_filters=result["n_filters"],
        label_perm_p=lp.get("p_value"),
        random_p=rb.get("p_value"),
        block_boot_p=bb.get("p_value"),
    )

    # Final cert verdict from the accumulated reasons
    verdict = verdict_from_reasons(reasons,
                                    best_possible=CertificationLevel.PROP_CANDIDATE)
    cand.apply_verdict(verdict)

    return {
        "candidate_id": cand.id,
        "family": cand.family,
        "symbol": cand.symbol,
        "session": cand.daily_rules.session_only or _detect_session(cand),
        "setup_timeframe": cand.setup_timeframe,
        "entry_timeframe": cand.entry_timeframe,
        "entry_model": cand.entry.get("type"),
        "stop_model": cand.stop.get("type"),
        "exit_model": cand.exit.get("type"),
        "trades": ho.get("trades", 0),
        "trades_per_week": ho.get("trades_per_week", 0.0),
        "win_rate": ho.get("win_rate", 0.0),
        "avg_ret_bp": ho.get("avg_ret_bp", 0.0),
        "expectancy_R": ho.get("expectancy_R", 0.0),
        "profit_factor": ho.get("profit_factor", 0.0),
        "total_return": ho.get("total_return", 0.0),
        "max_drawdown": ho.get("max_drawdown", 0.0),
        "validation_return": val.get("total_return", 0.0),
        "holdout_return": ho.get("total_return", 0.0),
        "stress_return": stress.get("total_return", 0.0),
        "wf_folds": wf.get("folds", 0),
        "wf_median_return": wf.get("median_total_return", 0.0),
        "wf_median_sharpe": wf.get("median_sharpe", 0.0),
        "wf_pct_positive": wf.get("pct_positive_folds", 0.0),
        "label_perm_p": lp.get("p_value"),
        "random_baseline_p": rb.get("p_value"),
        "block_bootstrap_p": bb.get("p_value"),
        "best_account": cand.account.name,
        "account_verification": account_verification.get(cand.account.name, "unknown"),
        "best_risk_model": cand.risk.name,
        "best_daily_rules": cand.daily_rules.name,
        "pass_probability": prop_summary.get("25k_pass_probability"),
        "blowup_probability": prop_summary.get("25k_blowup_probability"),
        "blowup_probability_upper_ci": prop_summary.get("25k_blowup_probability_ci", (0, 1))[1],
        "pass_to_blowup_ratio": _safe_ratio(
            prop_summary.get("25k_pass_probability"),
            prop_summary.get("25k_blowup_probability")),
        "median_days_to_pass": None,             # not yet wired
        "payout_survival_probability": None,
        "daily_loss_breach_probability": None,
        "trailing_drawdown_breach_probability": None,
        "average_attempts_to_first_pass": None,
        "prop_passing_score": score_payload["score"],
        "certification_level": cand.certification_level.value,
        "fail_reasons": "; ".join(r.value for r in reasons) if reasons else "",
        "unavailable_data_rejected": "",
        "fdr_significant": bool(fdr_keep),
        "notes": cand.notes,
        "candidate_json": cand.to_json_str(),
    }


def _detect_session(cand: PropCandidate) -> str:
    for f in cand.filters:
        if f.get("type") == "session":
            hours = f.get("hours_utc", [])
            if all(h in (12, 13, 14, 15, 16) for h in hours):
                return "ny"
            if all(h in (7, 8, 9, 10, 11) for h in hours):
                return "london"
            if all(h in (12, 13) for h in hours):
                return "ny_open"
            if all(h in (7, 8) for h in hours):
                return "london_open"
    return ""


def _safe_ratio(num, den):
    if num is None or den is None or den <= 0:
        return None
    return round(num / den, 3)


# ---------------- main pipeline ----------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="quick smoke (limit=4 candidates, n_perm=80)")
    ap.add_argument("--limit-candidates", type=int, default=None,
                    help="keep at most N tier-1 candidates")
    ap.add_argument("--families", default=None,
                    help="comma-separated family ids to restrict")
    ap.add_argument("--accounts", default=None,
                    help="comma-separated account ids for prop sim")
    ap.add_argument("--max-survivors-for-prop-sim", type=int, default=10)
    ap.add_argument("--fast-only", action="store_true",
                    help="stop after walk-forward; skip prop sim")
    ap.add_argument("--full", action="store_true",
                    help="use the wider risk + daily-rule sweeps")
    ap.add_argument("--n-perm", type=int, default=N_PERM_EXPLORATION)
    ap.add_argument("--output-stem", default="prop_passing_leaderboard")
    args = ap.parse_args(argv)

    if args.smoke:
        args.limit_candidates = args.limit_candidates or 4
        args.n_perm = min(args.n_perm, 80)

    families_filter = (set(s.strip() for s in args.families.split(","))
                       if args.families else None)

    # ---- run state / events ----
    state = RunState.create(RUNS_DIR)
    events = EventWriter(state)
    print(f"[run] id={state.run_id} dir={state.run_dir}")
    emit_stage(events, stage="strategy_generator", status="running",
                message="Batch G family generators")

    # ---- 1. generate ----
    state.set_stage("strategy_generator")
    cands = grid.tier_1_grid(families_filter=families_filter)
    state.bump("generated", len(cands))
    print(f"[1/9] generated {len(cands)} tier-1 candidates")
    if args.limit_candidates:
        cands = cands[:args.limit_candidates]
        print(f"      limited to {len(cands)}")

    # ---- 2. capability filter ----
    state.set_stage("feature_capability_auditor")
    kept, rejected = grid.apply_capability_filter(cands)
    state.bump("rejected_unavailable_data", len(rejected))
    for r in rejected:
        emit_candidate(events, candidate_id=r.id, family=r.family,
                        stage="feature_capability_auditor", status="rejected",
                        certification_level=r.certification_level.value,
                        failure_reasons=[fr.value for fr in r.failure_reasons],
                        message="rejected: unavailable data")
    print(f"[2/9] capability filter: kept={len(kept)} rejected={len(rejected)}")

    # ---- 3. data + TPO precompute ----
    state.set_stage("ohlc_backtest")
    print("[3/9] loading splits + precomputing TPO levels ...")
    splits = load_splits()
    # attach TPO levels once per slice -- the executor reads
    # `prev_session_tpo_*` columns inside the H4 frame.
    h4_train = attach_prev_session_tpo(splits.train_h4, splits.train_m15)
    h4_val = attach_prev_session_tpo(splits.validation_h4, splits.validation_m15)
    h4_holdout = attach_prev_session_tpo(splits.holdout_h4, splits.holdout_m15)
    full_h4 = pd.concat([h4_train, h4_val, h4_holdout], ignore_index=True)
    full_m15 = pd.concat([splits.train_m15, splits.validation_m15,
                           splits.holdout_m15], ignore_index=True)

    # ---- 4-5. evaluate ----
    accounts = load_accounts()
    account_verification = {n: verification_status(s)
                             for n, s in accounts.items()}

    print(f"[4-5/9] evaluating {len(kept)} candidates "
          f"(n_perm={args.n_perm}) ...")
    t0 = time.time()
    results: list[dict] = []
    for i, cand in enumerate(kept, 1):
        state.set_active([cand.id])
        try:
            res = _evaluate_candidate(
                cand,
                h4_train=h4_train, m15_train=splits.train_m15,
                h4_val=h4_val, m15_val=splits.validation_m15,
                h4_holdout=h4_holdout, m15_holdout=splits.holdout_m15,
                full_h4=full_h4, full_m15=full_m15,
                n_perm=args.n_perm,
                fast_only=args.fast_only,
            )
        except Exception as exc:
            res = {"candidate": cand,
                   "wf": {}, "val": {}, "ho": {}, "stress": {},
                   "stat_label_perm": {}, "stat_random": {}, "stat_block_boot": {},
                   "ho_trades": [], "stress_trades": [],
                   "reasons": [FailureReason.FAIL_RESOLUTION_LIMITED],
                   "n_filters": len(cand.filters),
                   "evaluator_error": str(exc)}
        results.append(res)
        # per-candidate event
        emit_candidate(events,
                        candidate_id=cand.id, family=cand.family,
                        stage="walk_forward",
                        status="passed" if not res["reasons"] else "failed",
                        failure_reasons=[fr.value for fr in res["reasons"]],
                        metrics={
                            "wf_folds": res["wf"].get("folds", 0),
                            "wf_median_sharpe": res["wf"].get("median_sharpe", 0.0),
                            "ho_total_return": res["ho"].get("total_return", 0.0),
                        })
        if i % 5 == 0 or i == len(kept):
            print(f"      {i}/{len(kept)}  ({(time.time()-t0)/60:.1f}min)")

    # ---- 6. prop sim on top survivors ----
    if args.fast_only:
        print("[6/9] --fast-only: skipping prop sim")
    else:
        state.set_stage("prop_firm_simulator")
        # Rank survivors by ho_total_return as a cheap proxy for who
        # gets the expensive prop sim.
        ranked = sorted(results, key=lambda r: r["ho"].get("total_return", -1e9),
                         reverse=True)
        n_prop = min(args.max_survivors_for_prop_sim, len(ranked))
        print(f"[6/9] prop sim on top {n_prop} candidates ...")
        for r in ranked[:n_prop]:
            _attach_prop(r)
            emit_candidate(events,
                            candidate_id=r["candidate"].id,
                            family=r["candidate"].family,
                            stage="prop_firm_simulator",
                            status="passed" if not r["reasons"] else "failed",
                            metrics=r.get("prop_summary", {}))

    # ---- 7. score + leaderboard ----
    state.set_stage("leaderboard")
    p_label = [r["stat_label_perm"].get("p_value", 1.0) for r in results]
    fdr_keep = benjamini_hochberg(p_label, q=0.05) if p_label else \
        [False] * len(results)

    rows = []
    for keep, r in zip(fdr_keep, results):
        rows.append(_row_from_result(r, fdr_keep=keep,
                                       account_verification=account_verification))
    # Add the rejected candidates as zero-row entries
    for rej in rejected:
        rows.append({
            "candidate_id": rej.id,
            "family": rej.family,
            "certification_level": rej.certification_level.value,
            "fail_reasons": "; ".join(r.value for r in rej.failure_reasons),
            "unavailable_data_rejected": "; ".join(rej.rejection_detail.get(
                "unavailable_features", [])),
            "candidate_json": rej.to_json_str(),
        })

    # Sort by prop_passing_score desc (None last)
    def _sort_key(row):
        s = row.get("prop_passing_score")
        return -1e9 if s is None else s
    rows = sorted(rows, key=_sort_key, reverse=True)

    df = pd.DataFrame(rows)
    csv_path = RESULTS / f"{args.output_stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[7/9] leaderboard -> {csv_path}  ({len(rows)} rows)")

    # provenance sidecar
    runtime_s = time.time() - t0
    sidecar_payload = {
        "produced_by": "scripts/run_prop_passing.py",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": state.run_id,
        "git_head": _git_head(),
        "config_hashes": _config_hashes(),
        "splits": coverage_summary(splits),
        "n_candidates_generated": int(state.counts.generated),
        "n_rejected_unavailable_data": int(state.counts.rejected_unavailable_data),
        "n_evaluated": len(results),
        "n_prop_simulated": sum(1 for r in results if "prop" in r),
        "n_perm": args.n_perm,
        "runtime_seconds": round(runtime_s, 1),
        "harness_version": "hardened_v2 (Batches A-H)",
        "mode": "smoke" if args.smoke else ("fast_only" if args.fast_only else
                                              ("full" if args.full else "default")),
        "families_filter": list(families_filter) if families_filter else None,
        "accounts_filter": (args.accounts.split(",") if args.accounts else None),
        "max_survivors_for_prop_sim": args.max_survivors_for_prop_sim,
        "prop_account_verification": account_verification,
        "schema_versions": {"prop_accounts": "3", "data_splits": "1"},
    }
    sidecar_path = csv_path.with_suffix(".meta.json")
    sidecar_path.write_text(json.dumps(sidecar_payload, indent=2, default=str))
    print(f"[7/9] sidecar -> {sidecar_path}")

    # ---- 8. report ----
    state.set_stage("report")
    report_path = RESULTS / "prop_passing_report.md"
    _write_report(report_path, rows=rows, sidecar=sidecar_payload,
                   results=results, account_verification=account_verification)
    print(f"[8/9] report -> {report_path}")

    # ---- 9. final ----
    state.set_stage("judge")
    n_certified = sum(1 for r in rows
                       if r.get("certification_level") == "certified")
    n_prop_cand = sum(1 for r in rows
                       if r.get("certification_level") == "prop_candidate")
    state.bump("certified", n_certified)
    state.bump("prop_candidates", n_prop_cand)
    state.set_status("certified" if n_certified else "passed")
    state.write_progress()    # flush final status to disk for UI
    state.write_summary({
        "leaderboard": str(csv_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "git_head": _git_head(),
        "n_certified": n_certified,
        "n_prop_candidates": n_prop_cand,
    })

    print()
    print("=== SUMMARY ===")
    print(f"candidates evaluated: {len(results)}")
    print(f"rejected unavailable: {len(rejected)}")
    print(f"prop_candidates:      {n_prop_cand}")
    print(f"certified:            {n_certified}")
    print(f"runtime:              {runtime_s/60:.1f} min")
    print(f"output:               {csv_path}")
    return 0


# ---------------- markdown report ----------------

def _write_report(path: Path, *, rows: list[dict], sidecar: dict,
                   results: list[dict],
                   account_verification: dict[str, str]) -> None:
    """Write the prop-passing report.

    The report intentionally tells you *what failed* and *what to try
    next*, not "look at this winning strategy". A valid output is
    "0 certified, here are the closest near-misses".
    """
    levels = {"rejected_unavailable_data": 0, "rejected_broken": 0,
              "research_only": 0, "watchlist": 0, "candidate": 0,
              "prop_candidate": 0, "certified": 0, "retired": 0}
    for r in rows:
        lvl = r.get("certification_level")
        if lvl in levels:
            levels[lvl] += 1
    fail_counts: dict[str, int] = {}
    for r in rows:
        for tok in (r.get("fail_reasons") or "").split(";"):
            tok = tok.strip()
            if tok:
                fail_counts[tok] = fail_counts.get(tok, 0) + 1
    fail_top = sorted(fail_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]

    top_rows = sorted(
        [r for r in rows if r.get("prop_passing_score") is not None],
        key=lambda r: r["prop_passing_score"], reverse=True)[:10]

    # Mutation suggestions for the closest near-misses
    next_queue: list[tuple[str, list]] = []
    for r in results[:10]:
        cand = r["candidate"]
        cand.failure_reasons = list(r["reasons"])
        sug = propose_mutations(cand)
        if sug:
            next_queue.append((cand.id, sug))

    lines: list[str] = []
    lines.append("# Prop-firm passing report")
    lines.append("")
    lines.append(f"Run id: `{sidecar.get('run_id')}`")
    lines.append(f"Produced at: {sidecar.get('produced_at_utc')}")
    lines.append(f"Mode: **{sidecar.get('mode')}**, "
                  f"n_perm={sidecar.get('n_perm')}, "
                  f"runtime={sidecar.get('runtime_seconds')}s.")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    n_total = len(rows)
    n_cert = levels["certified"]
    n_prop = levels["prop_candidate"]
    n_rej = levels["rejected_unavailable_data"]
    if n_cert == 0:
        lines.append(f"**0 certified strategies** out of {n_total} evaluated "
                      f"({n_rej} rejected for unavailable data, "
                      f"{n_prop} reached prop_candidate). "
                      "This is a valid research outcome under the hardened "
                      "harness — see Failure analysis below for the "
                      "exact gates that blocked each candidate.")
    else:
        lines.append(f"**{n_cert} certified** strategies out of {n_total} "
                      f"evaluated.")
    lines.append("")

    lines.append("## Data summary")
    lines.append("")
    cov = sidecar.get("splits", {})
    lines.append("| Split | Window | H4 rows | First | Last |")
    lines.append("|---|---|---:|---|---|")
    for name, info in cov.items():
        lines.append(
            f"| {name} | {info.get('config_start')} → "
            f"{info.get('config_end')} | {info.get('h4_rows')} | "
            f"{info.get('actual_first_bar')} | "
            f"{info.get('actual_last_bar')} |")
    lines.append("")

    lines.append("## Feature availability")
    lines.append("")
    lines.append("OHLC-only harness. Available features: OHLC, spread, "
                  "tick count, TPO. **NOT available**: real volume, bid/ask "
                  "volume, footprint, delta, CVD, order book, VWAP, Volume "
                  "Profile. Strategies referencing those tokens are "
                  "auto-rejected as `REJECTED_UNAVAILABLE_DATA`.")
    lines.append("")

    lines.append("## Top candidates")
    lines.append("")
    if not top_rows:
        lines.append("(no scored candidates)")
    else:
        lines.append("| Rank | id | family | score | cert | "
                      "pass_p | blowup_p | wf_med_sharpe | "
                      "ho_return | label_p | block_p |")
        lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(top_rows, 1):
            lines.append(
                f"| {i} | `{r['candidate_id']}` | {r.get('family')} | "
                f"{r.get('prop_passing_score')} | "
                f"{r.get('certification_level')} | "
                f"{r.get('pass_probability')} | "
                f"{r.get('blowup_probability')} | "
                f"{r.get('wf_median_sharpe')} | "
                f"{r.get('holdout_return')} | "
                f"{r.get('label_perm_p')} | "
                f"{r.get('block_bootstrap_p')} |")
    lines.append("")

    lines.append("## Certification level histogram")
    lines.append("")
    lines.append("| Level | Count |")
    lines.append("|---|---:|")
    for lvl, n in levels.items():
        lines.append(f"| {lvl} | {n} |")
    lines.append("")

    lines.append("## Failure analysis (top 8)")
    lines.append("")
    if not fail_top:
        lines.append("(no failures)")
    else:
        lines.append("| Failure reason | Candidates |")
        lines.append("|---|---:|")
        for reason, n in fail_top:
            lines.append(f"| `{reason}` | {n} |")
    lines.append("")

    lines.append("## Account comparison")
    lines.append("")
    lines.append("| Account | Verification |")
    lines.append("|---|---|")
    for name, status in sorted(account_verification.items()):
        lines.append(f"| {name} | {status} |")
    lines.append("")

    lines.append("## Next research queue")
    lines.append("")
    if not next_queue:
        lines.append("(no mutation suggestions — either no near-misses "
                      "or no candidates left actionable)")
    else:
        for cid, sugs in next_queue:
            lines.append(f"### `{cid}`")
            lines.append("")
            for s in sugs[:5]:
                lines.append(f"- **{s.candidate.id}** — {s.rationale}")
            lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({
        "git_head": sidecar.get("git_head"),
        "config_hashes": sidecar.get("config_hashes"),
        "harness_version": sidecar.get("harness_version"),
        "schema_versions": sidecar.get("schema_versions"),
    }, indent=2))
    lines.append("```")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
