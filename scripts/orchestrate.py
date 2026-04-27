"""Agentic search loop with walk-forward gating.

Pipeline:
    propose()  ->  walk_forward(spec, h4_long)  ->  holdout(spec, h4 + m15)
                                                    ->  certify()  ->  leaderboard.csv

The proposer here is a deterministic grid stub. Replace it later with a
Claude API call that reads `results/leaderboard.csv` and proposes the
next spec to try. The downstream pipeline does not change.

Certification rules (set in code below):
    median_sharpe         > 0
    pct_positive_folds   >= 0.55
    holdout_total_return  > 0
    holdout trades        >= 30   (refuse to certify on too few samples)
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


# ---------- proposer (placeholder grid) ----------

def propose() -> list[dict]:
    """Return a list of strategy specs to evaluate.

    This is a deterministic grid. Swap with an LLM proposer that reads
    the leaderboard and adapts; the rest of the pipeline is unchanged.
    """
    body_thresholds = [0.0, 0.5, 1.0]
    sessions: list[list[int] | None] = [
        None,
        [12, 16],            # NY hours UTC
        [8, 12, 16],         # London/NY
        [0, 4, 20],          # Asia/late
    ]
    regimes: list[dict | None] = [
        None,
        {"type": "regime", "ma_n": 50, "side": "with"},
        {"type": "regime", "ma_n": 50, "side": "against"},
    ]
    streaks = [None, 2]
    stops: list[dict] = [
        {"type": "none"},
        {"type": "h4_atr", "mult": 1.0, "atr_n": 14},
        {"type": "h4_atr", "mult": 2.0, "atr_n": 14},
    ]
    entries = ["m15_open", "m15_confirm", "m15_atr_stop"]

    specs: list[dict] = []
    for bt, sess, reg, k, stop, entry in product(
        body_thresholds, sessions, regimes, streaks, stops, entries
    ):
        filters: list[dict] = []
        if bt > 0:
            filters.append({"type": "body_atr", "min": bt, "atr_n": 14})
        if sess is not None:
            filters.append({"type": "session", "hours_utc": sess})
        if reg is not None:
            filters.append(reg)
        if k is not None:
            filters.append({"type": "min_streak", "k": k})

        spec = {
            "signal": {"type": "prev_color"},
            "filters": filters,
            "entry": {"type": entry},
            "stop": stop,
            "exit": {"type": "h4_close"},
            "cost_bps": 1.5,
        }
        spec["id"] = spec_id(spec)
        specs.append(spec)

    # de-dup by id
    seen = set()
    unique: list[dict] = []
    for s in specs:
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        unique.append(s)
    return unique


# ---------- critic ----------

def certify(wf: dict, ho: dict) -> bool:
    return (
        wf.get("folds", 0) >= 10
        and wf["median_sharpe"] > 0
        and wf["pct_positive_folds"] >= 0.55
        and ho["trades"] >= 30
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
    print(f"proposing {len(specs)} candidate strategies")

    rows = []
    fold_rows = []
    holdout_trade_rows = []
    for i, spec in enumerate(specs, 1):
        wf = walk_forward(spec, h4_long)
        ho_trades = run_full_sim(spec, h4, m15)
        ho = trades_to_metrics(spec["id"], ho_trades)

        ok = certify(wf, ho)
        rows.append({
            "id": spec["id"],
            "spec": json.dumps({k: v for k, v in spec.items() if k != "id"}),
            "wf_folds": wf["folds"],
            "wf_median_sharpe": wf["median_sharpe"],
            "wf_pct_positive_folds": wf["pct_positive_folds"],
            "wf_avg_total_return": wf["avg_total_return"],
            "wf_worst_fold_dd": wf["worst_fold_dd"],
            "ho_trades": ho["trades"],
            "ho_win_rate": ho["win_rate"],
            "ho_total_return": ho["total_return"],
            "ho_sharpe_ann": ho["sharpe_ann"],
            "ho_max_drawdown": ho["max_drawdown"],
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

        if i % 10 == 0 or i == len(specs):
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
                 "ho_trades", "ho_total_return", "ho_sharpe_ann", "certified"]
    print(df[show_cols].head(15).to_string(index=False))
    print(f"\nWrote: {LEADERBOARD}")


if __name__ == "__main__":
    run_search()
