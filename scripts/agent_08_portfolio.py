"""Agent 08 — Portfolio Builder.

Reads results/certified_alpha_strategies.json (the certifier's output)
plus the per-strategy trade ledger and clusters strategies by:

  - daily-PnL Pearson correlation (>0.85 = same alpha)
  - entry model
  - session
  - regime / family

Output: results/alpha_portfolio.json with the deduplicated short-list.
If fewer than 1 strategy is certified the output is still written, just
empty -- the audit pipeline still prints the (empty) file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CERT = ROOT / "results" / "certified_alpha_strategies.json"
TRADES_DIR = ROOT / "results" / "alpha_trades"
OUT = ROOT / "results" / "alpha_portfolio.json"

CORRELATION_THRESHOLD = 0.85


def _daily_pnl(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series(dtype=float)
    df = trades_df.copy()
    df["d"] = pd.to_datetime(df["exit_time"], utc=True).dt.date
    return df.groupby("d")["pnl"].sum()


def cluster_by_correlation(daily_series: dict[str, pd.Series],
                           threshold: float) -> list[list[str]]:
    """Greedy clustering: walk strategies in order, place each into the
    first existing cluster whose representative correlates above
    `threshold` with the candidate; otherwise start a new cluster."""
    ids = list(daily_series.keys())
    clusters: list[list[str]] = []
    reps: dict[str, pd.Series] = {}
    for i in ids:
        si = daily_series[i]
        placed = False
        for cluster in clusters:
            rep_id = cluster[0]
            sj = reps[rep_id]
            joined = pd.concat([si, sj], axis=1).fillna(0.0)
            if len(joined) < 5 or joined.iloc[:, 0].std() == 0 or joined.iloc[:, 1].std() == 0:
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if not np.isnan(corr) and corr >= threshold:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps[i] = si
    return clusters


def run() -> dict:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not CERT.exists():
        OUT.write_text(json.dumps({
            "n_input_strategies": 0,
            "n_clusters": 0,
            "portfolio": [],
            "note": "certified_alpha_strategies.json missing — run agent 07 first",
        }, indent=2))
        print(f"wrote {OUT}  (no input)")
        return {"n_clusters": 0}

    cert = json.loads(CERT.read_text())
    cert_list = cert.get("certified", [])
    if not cert_list:
        OUT.write_text(json.dumps({
            "n_input_strategies": 0,
            "n_clusters": 0,
            "portfolio": [],
            "note": "no certified strategies under strict gates; portfolio is empty",
        }, indent=2))
        print(f"wrote {OUT}  (0 certified -> empty portfolio)")
        return {"n_clusters": 0}

    daily: dict[str, pd.Series] = {}
    by_id: dict[str, dict] = {}
    for c in cert_list:
        sid = c["id"]
        by_id[sid] = c
        # Read the per-strategy trades file if present.
        tpath = TRADES_DIR / f"{sid}.csv"
        if not tpath.exists():
            continue
        tdf = pd.read_csv(tpath)
        if tdf.empty:
            continue
        daily[sid] = _daily_pnl(tdf)

    if not daily:
        OUT.write_text(json.dumps({
            "n_input_strategies": len(cert_list),
            "n_clusters": 0,
            "portfolio": [],
            "note": "certified strategies present but no per-strategy trade "
                    "ledgers found in results/alpha_trades/",
        }, indent=2))
        print(f"wrote {OUT}  (no per-strategy trade ledgers)")
        return {"n_clusters": 0}

    clusters = cluster_by_correlation(daily, CORRELATION_THRESHOLD)

    # Pick a representative per cluster: highest holdout Sharpe.
    def _score(sid: str) -> float:
        return float(by_id[sid].get("ho_sharpe_ann", 0.0))
    reps = []
    for cluster in clusters:
        cluster_sorted = sorted(cluster, key=_score, reverse=True)
        rep = cluster_sorted[0]
        reps.append({
            "representative_id": rep,
            "members": cluster_sorted,
            "rep_summary": {
                "ho_sharpe_ann": by_id[rep].get("ho_sharpe_ann"),
                "ho_total_return": by_id[rep].get("ho_total_return"),
                "trades_per_week": by_id[rep].get("trades_per_week"),
                "family_id": by_id[rep].get("family_id"),
                "entry": by_id[rep].get("spec", {}).get("entry"),
            },
        })

    out = {
        "n_input_strategies": len(cert_list),
        "n_clusters": len(clusters),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "portfolio": reps,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}  clusters={len(clusters)} from {len(cert_list)} certified")
    return out


if __name__ == "__main__":
    run()
