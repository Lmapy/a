"""End-to-end prop-challenge optimisation runner.

For every strategy that has a trade ledger in results/alpha_trades/
(plus any in results/refined_specs.csv), iterate over:

  account_models  × risk_models × daily_rule_sets

Run challenge MC + payout MC and emit the seven required output files.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prop_challenge.accounts import AccountSpec, load_all as load_accounts
from prop_challenge.challenge import run_challenge
from prop_challenge.lockout import DailyRules, all_rule_sets
from prop_challenge.payout import simulate_payout
from prop_challenge.risk import all_models as all_risk_models
from prop_challenge.score import passes_cert, prop_score

OUT = ROOT / "results"
TRADES_DIR = OUT / "alpha_trades"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- which strategies ----------

def collect_strategies() -> list[tuple[str, pd.DataFrame]]:
    """Return list of (strategy_id, trades_df) from the existing pipeline.

    Selection rule: by leaderboard wf_median_sharpe (the long-history
    edge signal), NOT by raw trade count. This keeps compression_breakout
    (1.11/wk, wf Sharpe 2.93) and fib_continuation (1.72/wk, wf 1.16)
    in the prop test even though they trade less than the high-frequency
    specs would.
    """
    LEADERBOARD = ROOT / "results" / "leaderboard.csv"
    out: list[tuple[str, pd.DataFrame]] = []
    if not TRADES_DIR.exists():
        return out
    # Build a wf_median_sharpe score map from the leaderboard if available.
    score_map: dict[str, float] = {}
    if LEADERBOARD.exists():
        try:
            lb = pd.read_csv(LEADERBOARD)
            for _, r in lb.iterrows():
                score_map[str(r["id"])] = float(r.get("wf_median_sharpe", 0.0) or 0.0)
        except Exception:
            score_map = {}
    for f in sorted(TRADES_DIR.glob("*.csv")):
        sid = f.stem
        df = pd.read_csv(f)
        if df.empty or len(df) < 30:
            continue
        out.append((sid, df))
    # Sort by leaderboard wf_median_sharpe desc; fall back to trade count.
    out.sort(key=lambda x: (-score_map.get(x[0], -1e9), -len(x[1])))
    return out[:TOP_N_STRATEGIES]


TOP_N_STRATEGIES = 6        # top-6 by walk-forward median Sharpe
# Lower MC counts to keep run-time interactive on the larger Dukascopy
# trade ledgers. Each MC bootstrap walks the full ledger; with 500-900
# trades per spec (vs 30-150 on the old broker) the per-sim cost rose
# 5-10x. 50/50 is enough resolution for the certify thresholds; bump
# to 200/200 for the final certified champion's report.
N_RUNS_CHALLENGE = 50
N_RUNS_PAYOUT    = 50


# ---------- selection focus ----------

ACCOUNT_FOCUS = ["topstep_50k", "mffu_50k", "generic_static_50k"]
RISK_FOCUS    = ["micro_1", "dollar_risk_50", "dollar_risk_100",
                 "pct_dd_buffer_2pct", "reduce_after_loss"]
RULES_FOCUS   = ["none", "max2", "stop_l2", "stop_w1", "ny_only_max2",
                 "dp500_dl300"]


def filter_named(items, names: list[str]):
    by_name = {x.name: x for x in items}
    return [by_name[n] for n in names if n in by_name]


# ---------- main ----------

def main() -> None:
    strategies = collect_strategies()
    if not strategies:
        raise SystemExit("no trade ledgers in results/alpha_trades/ — run scripts/run_alpha.py first")
    accounts = load_accounts()
    if not accounts:
        raise SystemExit("no accounts loaded from config/prop_accounts.json")

    risk_models_50k = filter_named(all_risk_models(accounts["topstep_50k"]), RISK_FOCUS)
    rules = filter_named(all_rule_sets(), RULES_FOCUS)
    focus_accounts = [accounts[k] for k in ACCOUNT_FOCUS if k in accounts]

    print(f"[prop] strategies={len(strategies)}  "
          f"accounts={len(focus_accounts)}  "
          f"risk_models={len(risk_models_50k)}  rule_sets={len(rules)}")

    rows_lb = []
    rows_acct = []
    rows_risk = []
    rows_rules = []
    rows_payout = []
    failure_modes: Counter = Counter()
    certified: list[dict] = []

    n_combos = len(strategies) * len(focus_accounts) * len(risk_models_50k) * len(rules)
    done = 0

    for sid, tdf in strategies:
        for acc in focus_accounts:
            risk_models = filter_named(all_risk_models(acc), RISK_FOCUS)
            best_score_for_strategy_account = -1e9
            best_combo = None
            for rm in risk_models:
                for rl in rules:
                    cr = run_challenge(tdf, acc, rm, rl, n_runs=N_RUNS_CHALLENGE)
                    pr = simulate_payout(tdf, acc, rm, rl, n_runs=N_RUNS_PAYOUT)
                    score = prop_score(cr, pr)
                    cert_ok, cert_fails = passes_cert(cr, pr)
                    row = {
                        "strategy_id": sid,
                        "account": acc.name,
                        "firm": acc.firm,
                        "drawdown_type": acc.drawdown_type,
                        "risk_model": rm.name,
                        "daily_rules": rl.name,
                        "pass_probability":  round(cr.pass_probability, 4),
                        "blowup_probability": round(cr.blowup_probability, 4),
                        "timeout_probability": round(cr.timeout_probability, 4),
                        "consistency_breach_rate": round(cr.consistency_breach_rate, 4),
                        "median_days_to_pass": cr.median_days_to_pass,
                        "median_end_balance": round(cr.median_end_balance, 2),
                        "first_payout_probability": round(pr.first_payout_probability, 4),
                        "blowup_before_payout_probability": round(pr.blowup_before_payout_probability, 4),
                        "median_days_to_payout": pr.median_days_to_payout,
                        "prop_score": score,
                        "cert_ok": cert_ok,
                        "cert_fails": "; ".join(cert_fails) if cert_fails else "",
                        "main_failure_reason": (
                            max(cr.breach_reason_histogram, key=cr.breach_reason_histogram.get)
                            if cr.breach_reason_histogram else "none"
                        ),
                    }
                    rows_lb.append(row)
                    for reason, count in cr.breach_reason_histogram.items():
                        failure_modes[reason] += count
                    if cert_ok:
                        certified.append(row)
                    if score > best_score_for_strategy_account:
                        best_score_for_strategy_account = score
                        best_combo = row
                    done += 1
            if best_combo is not None:
                rows_acct.append({**best_combo, "best_for_account": True})
        if (rows_lb and rows_lb[-1]):
            print(f"  evaluated {sid}: combos so far {done}/{n_combos}")

    # account comparison: best score per strategy × account
    pd.DataFrame(rows_lb).sort_values(
        ["cert_ok", "prop_score"], ascending=[False, False],
    ).to_csv(OUT / "prop_challenge_leaderboard.csv", index=False)

    pd.DataFrame(rows_acct).sort_values(
        ["account", "prop_score"], ascending=[True, False],
    ).to_csv(OUT / "prop_account_comparison.csv", index=False)

    # Risk model comparison: aggregate by risk_model
    df_lb = pd.DataFrame(rows_lb)
    risk_summary = df_lb.groupby("risk_model").agg(
        n=("prop_score", "size"),
        median_pass=("pass_probability", "median"),
        median_blowup=("blowup_probability", "median"),
        median_payout=("first_payout_probability", "median"),
        median_score=("prop_score", "median"),
    ).round(4).reset_index().sort_values("median_score", ascending=False)
    risk_summary.to_csv(OUT / "prop_risk_model_comparison.csv", index=False)

    rules_summary = df_lb.groupby("daily_rules").agg(
        n=("prop_score", "size"),
        median_pass=("pass_probability", "median"),
        median_blowup=("blowup_probability", "median"),
        median_payout=("first_payout_probability", "median"),
        median_score=("prop_score", "median"),
    ).round(4).reset_index().sort_values("median_score", ascending=False)
    rules_summary.to_csv(OUT / "prop_daily_rules_comparison.csv", index=False)

    # Payout survival
    payout_summary = df_lb.groupby(["strategy_id", "account"]).agg(
        median_payout_prob=("first_payout_probability", "median"),
        max_payout_prob=("first_payout_probability", "max"),
        median_blowup_pre_payout=("blowup_before_payout_probability", "median"),
        median_days_to_payout=("median_days_to_payout", "median"),
    ).round(4).reset_index().sort_values(
        ["max_payout_prob", "median_payout_prob"], ascending=[False, False])
    payout_summary.to_csv(OUT / "prop_payout_survival.csv", index=False)

    # Failure modes
    failure_doc = {
        "n_combos_evaluated": len(rows_lb),
        "n_certified": len(certified),
        "breach_histogram": dict(failure_modes),
        "top_5_failure_modes": failure_modes.most_common(5),
    }
    (OUT / "prop_failure_modes.json").write_text(json.dumps(failure_doc, indent=2))

    # Certified
    (OUT / "certified_prop_challenge_strategies.json").write_text(json.dumps({
        "policy": "Strategy + risk + daily_rules combo certifies if "
                  "pass_probability >= 0.35, first_payout_probability >= 0.20, "
                  "blowup_probability <= 0.15, consistency_breach_rate <= 0.15, "
                  "median_days_to_pass <= 30. Tested across at least 2 account "
                  "models is verified by the leaderboard.",
        "n_certified": len(certified),
        "certified": certified,
    }, indent=2))

    print()
    print(f"=== prop challenge complete: {len(rows_lb)} combos / "
          f"{len(certified)} certified ===")
    print(f"  outputs: {OUT}/prop_challenge_leaderboard.csv, "
          f"prop_account_comparison.csv, prop_risk_model_comparison.csv, "
          f"prop_daily_rules_comparison.csv, prop_payout_survival.csv, "
          f"certified_prop_challenge_strategies.json, prop_failure_modes.json")


if __name__ == "__main__":
    main()
