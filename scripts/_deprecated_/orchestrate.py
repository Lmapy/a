"""Agentic search loop with walk-forward gating + 3-5 trades/week constraint.

Pipeline:
    propose() -> walk_forward(spec, h4_long) -> holdout(spec, h4 + m15)
              -> certify() -> leaderboard.csv

The proposer here is a deterministic grid stub. Replace it later with a
Claude API call that reads `results/leaderboard.csv` and proposes the
next spec to try. The downstream pipeline does not change.

The grid covers two strategy families:

  A. CONTINUATION (existing) -- enter on the new H4 candle, in the
     direction of the previous H4 candle, with M15 timing variants.
  B. RETRACEMENT (new) -- as A, but wait for price to pull back into
     the previous H4 candle (50% midpoint), with structural stops at
     prev-H4 open or low/high, and TP at the opposite prev-H4 extreme.

Certification rule (set in code below):
    walk-forward folds       >= 10
    walk-forward median Sharpe > 0
    walk-forward pct positive folds >= 0.55
    3.0 <= holdout trades-per-week <= 5.0
    holdout total return > 0
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from strategy import (  # noqa: E402
    run_full_sim, run_h4_sim, spec_id, trades_to_metrics,
)
from walkforward import walk_forward  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results"

H4_LONG = DATA / "XAUUSD_H4_long.csv"
H4_MATCHED = DATA / "XAUUSD_H4_matched.csv"
M15_MATCHED = DATA / "XAUUSD_M15_matched.csv"

LEADERBOARD = OUT / "leaderboard.csv"
HOLDOUT_TRADES = OUT / "search_holdout_trades.csv"
SEARCH_FOLDS = OUT / "search_folds.csv"


# Holdout window: 2026-01-30 -> 2026-04-01 = 62 days ≈ 8.86 weeks.
HOLDOUT_WEEKS = 8.86


# ---------- proposer (placeholder grid) ----------

def _continuation_grid() -> list[dict]:
    body_thresholds = [None, 0.5, 1.0]
    sessions: list[list[int] | None] = [
        None,
        [12, 16],
        [8, 12, 16],
        [0, 4],
    ]
    regimes: list[dict | None] = [
        None,
        {"type": "regime", "ma_n": 50, "side": "with"},
    ]
    streaks = [None, 2]
    stops: list[dict] = [
        {"type": "none"},
        {"type": "h4_atr", "mult": 1.0, "atr_n": 14},
        {"type": "prev_h4_open"},
        {"type": "prev_h4_extreme"},
    ]
    entries = ["m15_open", "m15_confirm", "m15_atr_stop"]
    exits = [{"type": "h4_close"}, {"type": "prev_h4_extreme_tp"}]

    out: list[dict] = []
    for bt, sess, reg, k, stop, entry, ex in product(
        body_thresholds, sessions, regimes, streaks, stops, entries, exits,
    ):
        # Skip senseless combo: m15_atr_stop entry already implies its own stop
        # so pair only with stop=none to keep semantics clean.
        if entry == "m15_atr_stop" and stop["type"] != "none":
            continue
        filters: list[dict] = []
        if bt is not None:
            filters.append({"type": "body_atr", "min": bt, "atr_n": 14})
        if sess is not None:
            filters.append({"type": "session", "hours_utc": sess})
        if reg is not None:
            filters.append(reg)
        if k is not None:
            filters.append({"type": "min_streak", "k": k})
        out.append({
            "signal": {"type": "prev_color"},
            "filters": filters,
            "entry": {"type": entry},
            "stop": stop,
            "exit": ex,
            "cost_model": "spread",
            "cost_bps": 1.5,
        })
    return out


def _retracement_grid() -> list[dict]:
    body_thresholds = [None, 0.5]
    sessions: list[list[int] | None] = [
        None,
        [12, 16],
        [8, 12, 16],
        [0, 4],
    ]
    regimes: list[dict | None] = [
        None,
        {"type": "regime", "ma_n": 50, "side": "with"},
    ]
    classes: list[list[str] | None] = [None, ["trend"], ["trend", "rotation"]]
    stops: list[dict] = [
        {"type": "prev_h4_open"},
        {"type": "prev_h4_extreme"},
    ]
    exits = [{"type": "prev_h4_extreme_tp"}, {"type": "h4_close"}]
    # Fib retracement levels searched (0.5 = midpoint = the user's original idea).
    fib_levels = [0.382, 0.5, 0.618, 0.786]

    out: list[dict] = []
    for bt, sess, reg, cls, stop, ex, lvl in product(
        body_thresholds, sessions, regimes, classes, stops, exits, fib_levels,
    ):
        filters: list[dict] = []
        if bt is not None:
            filters.append({"type": "body_atr", "min": bt, "atr_n": 14})
        if sess is not None:
            filters.append({"type": "session", "hours_utc": sess})
        if reg is not None:
            filters.append(reg)
        if cls is not None:
            filters.append({"type": "candle_class", "classes": cls})
        out.append({
            "signal": {"type": "prev_color"},
            "filters": filters,
            "entry": {"type": "m15_retrace_fib", "level": lvl},
            "stop": stop,
            "exit": ex,
            "cost_model": "spread",
            "cost_bps": 1.5,
        })
    return out


def propose() -> list[dict]:
    specs = _continuation_grid() + _retracement_grid()
    seen = set()
    unique: list[dict] = []
    for s in specs:
        s["id"] = spec_id(s)
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        unique.append(s)
    return unique


# ---------- critic ----------

def certify(wf: dict, ho: dict, tpw: float) -> bool:
    return (
        wf.get("folds", 0) >= 10
        and wf["median_sharpe"] > 0
        and wf["pct_positive_folds"] >= 0.55
        and 3.0 <= tpw <= 5.0
        and ho["total_return"] > 0
    )


# ---------- main loop ----------

def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, parse_dates=["time"]).sort_values("time").reset_index(drop=True)


def run_search() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h4_long = load_csv(H4_LONG)
    h4 = load_csv(H4_MATCHED)
    m15 = load_csv(M15_MATCHED)

    specs = propose()
    print(f"proposing {len(specs)} candidate strategies "
          f"({len(_continuation_grid())} continuation + {len(_retracement_grid())} retracement)")

    rows = []
    fold_rows = []
    holdout_trade_rows = []
    for i, spec in enumerate(specs, 1):
        wf = walk_forward(spec, h4_long)
        ho_trades, ho_diag = run_full_sim(spec, h4, m15, return_diag=True)
        ho = trades_to_metrics(spec["id"], ho_trades)
        tpw = ho["trades"] / HOLDOUT_WEEKS

        ok = certify(wf, ho, tpw)
        rows.append({
            "id": spec["id"],
            "spec": json.dumps({k: v for k, v in spec.items() if k != "id"}),
            "wf_folds": wf["folds"],
            "wf_median_sharpe": wf["median_sharpe"],
            "wf_pct_positive_folds": wf["pct_positive_folds"],
            "wf_avg_total_return": wf["avg_total_return"],
            "wf_worst_fold_dd": wf["worst_fold_dd"],
            "ho_trades": ho["trades"],
            "ho_trades_per_week": round(tpw, 2),
            "ho_win_rate": ho["win_rate"],
            "ho_total_return": ho["total_return"],
            "ho_sharpe_ann": ho["sharpe_ann"],
            "ho_max_drawdown": ho["max_drawdown"],
            "ho_diag_signal_zero": ho_diag.get("signal_zero", 0),
            "ho_diag_no_retrace": ho_diag.get("no_retrace", 0),
            "ho_diag_no_confirm": ho_diag.get("no_confirm", 0),
            "ho_diag_no_subbars": ho_diag.get("no_subbars", 0),
            "ho_diag_missing_prev_levels": ho_diag.get("missing_prev_levels", 0),
            "certified": ok,
        })
        if isinstance(wf.get("fold_table"), pd.DataFrame) and not wf["fold_table"].empty:
            ft = wf["fold_table"].copy()
            ft.insert(0, "id", spec["id"])
            fold_rows.append(ft)
        if ho_trades:
            tdf = pd.DataFrame([{
                "id": spec["id"],
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "dir": t.direction, "entry": t.entry, "exit": t.exit,
                "cost": t.cost, "ret": t.ret,
            } for t in ho_trades])
            holdout_trade_rows.append(tdf)

        if i % 25 == 0 or i == len(specs):
            print(f"  evaluated {i}/{len(specs)}")

    df = pd.DataFrame(rows).sort_values(
        ["certified", "wf_median_sharpe", "ho_total_return"],
        ascending=[False, False, False],
    )
    df.to_csv(LEADERBOARD, index=False)

    if fold_rows:
        pd.concat(fold_rows, ignore_index=True).to_csv(SEARCH_FOLDS, index=False)
    if holdout_trade_rows:
        pd.concat(holdout_trade_rows, ignore_index=True).to_csv(HOLDOUT_TRADES, index=False)

    n_cert = int(df["certified"].sum()) if len(df) else 0
    print()
    print(f"=== search complete: {n_cert} certified out of {len(df)} ===")
    show_cols = ["id", "wf_folds", "wf_median_sharpe", "wf_pct_positive_folds",
                 "ho_trades", "ho_trades_per_week", "ho_total_return",
                 "ho_sharpe_ann", "certified"]
    print(df[show_cols].head(20).to_string(index=False))
    print(f"\nWrote: {LEADERBOARD}")


if __name__ == "__main__":
    run_search()
