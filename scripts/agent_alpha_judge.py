"""Alpha Judge — meta-agent.

Reads every result file the rest of the pipeline produced and writes a
single structured verdict at results/alpha_judge.json:

  {
    "verdict": "no_alpha_certified",
    "evaluated": 16,
    "passes_critic": 3,
    "certified": 0,
    "common_failures": [
      "edge disappears under shuffled-outcome test",
      "random-baseline p-value above 0.05",
      "profit concentrated in <2 sessions",
      ...
    ],
    "recommendations": [
      "longer M15 history is the single biggest unlock",
      "add session filter (NY-only) to reduce concentration risk",
      ...
    ],
    "near_misses": [ ... up to 3 closest candidates with reasons ... ]
  }

The judge is deliberately deterministic — it reads structured outputs,
applies pattern-matching rules, and emits an actionable summary. No
LLM call. The narrative comes from rules over data, not generation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LB = ROOT / "results" / "leaderboard.csv"
CRITIC = ROOT / "results" / "critic_report.json"
CERT = ROOT / "results" / "certified_alpha_strategies.json"
PROP = ROOT / "results" / "prop_simulation.json"
OUT = ROOT / "results" / "alpha_judge.json"


# ---------- pattern detection over the leaderboard ----------

def detect_patterns(lb: pd.DataFrame) -> dict:
    out: dict = {}
    if lb.empty:
        return {"empty": True}

    out["n_evaluated"] = int(len(lb))
    out["n_passes_critic"] = int(lb["passes_critic"].astype(str).str.lower().eq("true").sum())
    out["n_certified"] = int(lb["certified"].astype(str).str.lower().eq("true").sum())

    # 1. Edge dies after spread / stress
    stress_pos = (lb["ho_total_return"] > 0) & (lb["stress_total_return"] > 0)
    stress_kills = (lb["ho_total_return"] > 0) & (lb["stress_total_return"] <= 0)
    out["edge_dies_under_stress"] = int(stress_kills.sum())
    out["edge_survives_stress"]   = int(stress_pos.sum())

    # 2. Statistical-significance failures
    # high p-values mean the strategy looks like random ordering / random direction
    out["shuffle_p_above_0.05"] = int((lb["shuffle_p_value"] > 0.05).sum())
    out["random_p_above_0.05"]  = int((lb["random_p_value"]  > 0.05).sum())

    # 3. wf positive-folds bottleneck
    out["wf_below_60pct_positive_folds"] = int((lb["wf_pct_positive_folds"] < 0.60).sum())

    # 4. holdout-trade-count thrashing
    out["below_3_trades_per_week"] = int((lb["ho_trades_per_week"] < 3).sum())
    out["above_5_trades_per_week"] = int((lb["ho_trades_per_week"] > 5).sum())

    # 5. profit-factor distribution
    out["pf_above_1.2"] = int((lb["ho_profit_factor"] > 1.2).sum())
    out["pf_below_1.0"] = int((lb["ho_profit_factor"] < 1.0).sum())

    return out


# ---------- critic-failure clustering ----------

def critic_failure_histogram() -> dict[str, int]:
    if not CRITIC.exists():
        return {}
    crit = json.loads(CRITIC.read_text())
    counts: dict[str, int] = {}
    for r in crit.get("critic_results", []):
        for reason in r.get("failure_reasons", []) or []:
            head = reason.split("(")[0].strip().rstrip(":").strip()
            counts[head] = counts.get(head, 0) + 1
    return counts


# ---------- near-miss extraction ----------

def near_misses(lb: pd.DataFrame, n: int = 3) -> list[dict]:
    cands = lb[lb["passes_critic"].astype(str).str.lower() == "true"].copy()
    if cands.empty:
        cands = lb.copy()
    cands = cands.sort_values(
        ["certified", "wf_median_sharpe", "ho_total_return"],
        ascending=[False, False, False],
    ).head(n)
    out = []
    for _, r in cands.iterrows():
        out.append({
            "id": r["id"],
            "family_id": r.get("family_id"),
            "wf_folds": int(r.get("wf_folds", 0)),
            "wf_median_sharpe": float(r.get("wf_median_sharpe", 0.0)),
            "wf_pct_positive_folds": float(r.get("wf_pct_positive_folds", 0.0)),
            "ho_trades_per_week": float(r.get("ho_trades_per_week", 0.0)),
            "ho_total_return": float(r.get("ho_total_return", 0.0)),
            "ho_profit_factor": float(r.get("ho_profit_factor", 0.0)),
            "ho_sharpe_ann": float(r.get("ho_sharpe_ann", 0.0)),
            "passes_critic": bool(str(r.get("passes_critic")).lower() == "true"),
            "certified": bool(str(r.get("certified")).lower() == "true"),
            "cert_failures": r.get("cert_failures", ""),
            "critic_failures": r.get("critic_failures", ""),
            "stress_total_return": float(r.get("stress_total_return", 0.0)),
            "shuffle_p_value": float(r.get("shuffle_p_value", 1.0)),
            "random_p_value": float(r.get("random_p_value", 1.0)),
        })
    return out


# ---------- recommendation engine ----------

def build_recommendations(patterns: dict, critic_hist: dict[str, int]) -> list[str]:
    recs: list[str] = []
    n = patterns.get("n_evaluated", 0)
    if n == 0:
        return ["no strategies evaluated; run scripts/run_alpha.py first"]

    if patterns.get("n_certified", 0) == 0:
        recs.append(
            "no strategy survived the strict cert + critic + FDR gates; "
            "the closest misses are listed in near_misses below."
        )

    # statistical sig
    if patterns.get("shuffle_p_above_0.05", 0) == n and n > 0:
        recs.append(
            "shuffle test fails on every spec — the per-trade returns "
            "carry no time-structure signal beyond their marginal "
            "distribution; longer M15 history is needed before edges "
            "can be statistically distinguished from random ordering."
        )
    if patterns.get("random_p_above_0.05", 0) >= max(1, int(n * 0.9)):
        recs.append(
            "random-baseline p-value above 0.05 on most specs — at this "
            "trade frequency a random direction sequence reproduces a "
            "similar Sharpe; selectivity needs to rise OR sample size."
        )

    # wf positive-folds
    if patterns.get("wf_below_60pct_positive_folds", 0) >= max(1, int(n * 0.7)):
        recs.append(
            ">=70% of specs fall short of the 60% positive-fold gate; "
            "the underlying edge is regime-dependent — try regime_class "
            "filters (trend-only vs range-only) to focus the signal."
        )

    # frequency band
    if patterns.get("below_3_trades_per_week", 0) >= max(1, int(n * 0.3)):
        recs.append(
            "many specs trade <3 times/week; loosen filters or pick a "
            "less restrictive entry model to bring trade count up."
        )
    if patterns.get("above_5_trades_per_week", 0) >= max(1, int(n * 0.3)):
        recs.append(
            "many specs trade >5/week; tighten filters (body_atr.min "
            "to 0.7-1.0, add session, add streak) to reduce noise trades."
        )

    # stress kills
    if patterns.get("edge_dies_under_stress", 0) > 0:
        recs.append(
            f"{patterns['edge_dies_under_stress']} spec(s) flip negative "
            "under spread×1.5 / slippage×2 — those edges are not "
            "real-world tradable; reduce trade frequency or use limit "
            "orders so spread cost is amortised."
        )

    # critic failures
    if critic_hist:
        top_critic = sorted(critic_hist.items(), key=lambda kv: -kv[1])[:3]
        for label, count in top_critic:
            recs.append(f"critic failure '{label}' appears {count} time(s) "
                        "across the leaderboard; investigate the trade ledger")

    # blanket recommendations
    recs.append(
        "single biggest unlock: more *same-broker* M5 history. Sub-M15 "
        "execution-model improvements (sweep_reclaim, "
        "minor_structure_break) cannot be fairly tested until that "
        "lands. See results/data_manifest.json -> known_gaps."
    )
    return recs


# ---------- main ----------

def run() -> dict:
    if not LB.exists():
        out = {"verdict": "no_results", "note": "leaderboard.csv missing — run scripts/run_alpha.py first"}
        OUT.write_text(json.dumps(out, indent=2))
        return out

    lb = pd.read_csv(LB)
    patterns = detect_patterns(lb)
    critic_hist = critic_failure_histogram()
    nms = near_misses(lb, n=3)

    n_cert = patterns.get("n_certified", 0)
    if n_cert == 0:
        verdict = "no_alpha_certified"
        verdict_detail = (
            "0 strategies passed all gates. Closest misses are listed "
            "below with the specific gates that rejected them."
        )
    elif n_cert <= 2:
        verdict = "alpha_candidate_with_low_confidence"
        verdict_detail = (
            f"{n_cert} strategies passed all gates, but the count is "
            "small relative to the search size — verify with longer "
            "out-of-sample data before live deployment."
        )
    else:
        verdict = "multiple_alpha_candidates"
        verdict_detail = (
            f"{n_cert} strategies passed all gates; cluster them via "
            "agent_08_portfolio.py and select diversified survivors."
        )

    out = {
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "patterns": patterns,
        "critic_failure_histogram": critic_hist,
        "common_failures": [k for k, _ in sorted(
            critic_hist.items(), key=lambda kv: -kv[1])[:5]],
        "recommendations": build_recommendations(patterns, critic_hist),
        "near_misses": nms,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"  alpha_judge: verdict={verdict}  "
          f"(certified={n_cert}, near_misses={len(nms)})")
    return out


if __name__ == "__main__":
    run()
