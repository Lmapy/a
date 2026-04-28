"""Skeptic agent — find what we missed, where the edge is fragile, and
which filter is doing the work.

For each top certified champion in the leaderboard, runs three probes:

  1. PERTURBATION  — single-knob nudges to cost_bps, atr_n, ma_n, fib
                     level, stop multiplier. Reveals how fragile the
                     certification is. The metric of interest is "still
                     certified" under each perturbation.

  2. ATTRIBUTION   — drop each filter one at a time, re-run walk-forward.
                     The Sharpe drop tells us which filter is load-bearing.

  3. COVERAGE      — generate "near-miss" variants the original grid did
                     not enumerate (body_atr at 0.7, ma_n at 80, fib level
                     0.5 with streak2, etc.) and evaluate them. Surfaces
                     specs the search would have missed.

Output: results/skeptic.csv  (one row per probe variant)
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from strategy import run_full_sim, spec_id, trades_to_metrics  # noqa: E402
from walkforward import walk_forward  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results"

LEADERBOARD = OUT / "leaderboard.csv"
SKEPTIC = OUT / "skeptic.csv"

# Use the orchestrator's certification rule verbatim so this is apples-to-apples.
TRADES_PER_WEEK_MIN = 3.0
TRADES_PER_WEEK_MAX = 5.0


def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, parse_dates=["time"]).sort_values("time").reset_index(drop=True)


def certify(wf: dict, ho: dict) -> bool:
    return (
        wf.get("folds", 0) >= 10
        and wf["median_sharpe"] > 0
        and wf["pct_positive_folds"] >= 0.55
        and ho["trades"] >= 30
        and ho["total_return"] > 0
        and TRADES_PER_WEEK_MIN <= ho.get("trades_per_week", 0) <= TRADES_PER_WEEK_MAX
    )


def evaluate(spec: dict, h4_long: pd.DataFrame, h4: pd.DataFrame, m15: pd.DataFrame) -> dict:
    spec = deepcopy(spec)
    spec.setdefault("id", spec_id(spec))
    wf = walk_forward(spec, h4_long)
    ho_trades = run_full_sim(spec, h4, m15)
    ho = trades_to_metrics(spec["id"], ho_trades)
    weeks = (h4["time"].iloc[-1] - h4["time"].iloc[0]).total_seconds() / 604_800.0
    ho["trades_per_week"] = round(ho["trades"] / weeks, 2) if weeks > 0 else 0.0
    ok = certify(wf, ho)
    return {
        "spec_id": spec["id"],
        "spec_json": json.dumps({k: v for k, v in spec.items() if k != "id"}),
        "wf_folds": wf["folds"],
        "wf_median_sharpe": wf["median_sharpe"],
        "wf_pct_positive_folds": wf["pct_positive_folds"],
        "wf_avg_total_return": wf["avg_total_return"],
        "ho_trades": ho["trades"],
        "ho_trades_per_week": ho["trades_per_week"],
        "ho_total_return": ho["total_return"],
        "ho_sharpe_ann": ho["sharpe_ann"],
        "still_certified": bool(ok),
    }


# ---------- probe generators ----------

def _set_filter_param(spec: dict, ftype: str, key: str, value) -> dict:
    new = deepcopy(spec)
    found = False
    for f in new["filters"]:
        if f["type"] == ftype:
            f[key] = value
            found = True
            break
    if not found:
        # add the filter if it wasn't there
        if ftype == "body_atr":
            new["filters"].append({"type": "body_atr", "min": value, "atr_n": 14})
        elif ftype == "regime":
            new["filters"].append({"type": "regime", "ma_n": value, "side": "with"})
        elif ftype == "min_streak":
            new["filters"].append({"type": "min_streak", "k": value})
    return new


def _drop_filter(spec: dict, ftype: str) -> dict:
    new = deepcopy(spec)
    new["filters"] = [f for f in new["filters"] if f["type"] != ftype]
    return new


def perturbations(spec: dict) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []

    # cost sensitivity (broker spread is real; how much head-room before edge dies?)
    for c in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        new = deepcopy(spec)
        new["cost_bps"] = c
        out.append((f"cost_bps={c}", new))

    # body_atr.min
    body_filter = next((f for f in spec.get("filters", []) if f["type"] == "body_atr"), None)
    if body_filter is not None:
        for v in [0.0, 0.3, 0.5, 0.7, 0.9, 1.2]:
            if abs(v - float(body_filter.get("min", 0.5))) < 1e-9:
                continue
            out.append((f"body_atr.min={v}", _set_filter_param(spec, "body_atr", "min", v)))
        for v in [7, 10, 14, 21]:
            if v == int(body_filter.get("atr_n", 14)):
                continue
            out.append((f"body_atr.atr_n={v}", _set_filter_param(spec, "body_atr", "atr_n", v)))

    # regime.ma_n
    regime_filter = next((f for f in spec.get("filters", []) if f["type"] == "regime"), None)
    if regime_filter is not None:
        for v in [20, 30, 50, 80, 100, 150]:
            if v == int(regime_filter.get("ma_n", 50)):
                continue
            out.append((f"regime.ma_n={v}", _set_filter_param(spec, "regime", "ma_n", v)))

    # fib level
    if spec["entry"].get("type") == "m15_retrace_fib":
        cur = float(spec["entry"].get("level", 0.5))
        for v in [0.236, 0.382, 0.5, 0.618, 0.786]:
            if abs(v - cur) < 1e-9:
                continue
            new = deepcopy(spec)
            new["entry"]["level"] = v
            out.append((f"fib_level={v}", new))

    # stop / TP swap
    stop_alts: list[dict] = [
        {"type": "none"},
        {"type": "prev_h4_open"},
        {"type": "prev_h4_extreme"},
        {"type": "h4_atr", "mult": 1.0, "atr_n": 14},
    ]
    for s in stop_alts:
        if s == spec["stop"]:
            continue
        new = deepcopy(spec); new["stop"] = s
        label = "stop=" + s["type"] + (f"x{s.get('mult', 1.0)}" if s["type"] == "h4_atr" else "")
        out.append((label, new))

    return out


def attribution(spec: dict) -> list[tuple[str, dict]]:
    """Drop one filter at a time so we can attribute Sharpe contribution."""
    out: list[tuple[str, dict]] = []
    types = sorted({f["type"] for f in spec.get("filters", [])})
    for ftype in types:
        out.append((f"drop_{ftype}", _drop_filter(spec, ftype)))
    if len(types) >= 2:
        # also try dropping all filters (raw signal only)
        new = deepcopy(spec); new["filters"] = []
        out.append(("drop_ALL_filters", new))
    return out


def coverage(spec: dict) -> list[tuple[str, dict]]:
    """Near-miss variants the search grid did not enumerate."""
    out: list[tuple[str, dict]] = []

    # ma_n values not in the original grid
    has_regime = any(f["type"] == "regime" for f in spec.get("filters", []))
    if has_regime:
        for v in [30, 80, 100]:  # original grid only used 50
            out.append((f"coverage_ma_n={v}", _set_filter_param(spec, "regime", "ma_n", v)))

    # body_atr.min off-grid
    has_body = any(f["type"] == "body_atr" for f in spec.get("filters", []))
    if has_body:
        for v in [0.3, 0.7]:  # original grid only used 0.5 and 1.0
            out.append((f"coverage_body_atr.min={v}", _set_filter_param(spec, "body_atr", "min", v)))

    # add streak filter if not present
    if not any(f["type"] == "min_streak" for f in spec.get("filters", [])):
        for k in [2, 3]:
            out.append((f"coverage_add_streak{k}", _set_filter_param(spec, "min_streak", "k", k)))

    # add regime filter if not present
    if not has_regime:
        out.append(("coverage_add_regime50with", _set_filter_param(spec, "regime", "ma_n", 50)))

    # fib + streak combination (not in original retracement grid)
    if spec["entry"].get("type") == "m15_retrace_fib":
        if not any(f["type"] == "min_streak" for f in spec.get("filters", [])):
            new = _set_filter_param(spec, "min_streak", "k", 2)
            out.append(("coverage_fib_plus_streak2", new))

    return out


# ---------- main ----------

def pick_champions(df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    cert = df[df["certified"].astype(str).str.lower() == "true"].copy()
    if cert.empty:
        return []

    def signature(row) -> str:
        spec = json.loads(row["spec"])
        sig_parts = []
        for f in spec.get("filters", []):
            sig_parts.append(json.dumps(f, sort_keys=True))
        sig_parts.append(spec["entry"]["type"])
        if spec["entry"].get("type") == "m15_retrace_fib":
            sig_parts.append(f"lvl={spec['entry'].get('level')}")
        return "||".join(sig_parts)

    cert["signature"] = cert.apply(signature, axis=1)
    cert = cert.sort_values(["wf_median_sharpe", "ho_total_return"], ascending=False)
    seen = set()
    keep = []
    for _, row in cert.iterrows():
        if row["signature"] in seen:
            continue
        seen.add(row["signature"])
        spec = json.loads(row["spec"])
        spec["id"] = row["id"]
        keep.append(spec)
        if len(keep) >= top_n:
            break
    return keep


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not LEADERBOARD.exists():
        raise SystemExit("results/leaderboard.csv missing — run `make search` first")
    lb = pd.read_csv(LEADERBOARD)

    h4_long = load_csv(DATA / "XAUUSD_H4_long.csv")
    h4 = load_csv(DATA / "XAUUSD_H4_matched.csv")
    m15 = load_csv(DATA / "XAUUSD_M15_matched.csv")

    champions = pick_champions(lb, top_n=5)
    if not champions:
        print("no certified champions found; nothing to skeptic")
        return
    print(f"running skeptic on {len(champions)} certified champion(s)")

    rows: list[dict] = []
    for champ in champions:
        cid = champ["id"]
        baseline = evaluate(champ, h4_long, h4, m15)
        baseline.update({"champion_id": cid, "probe_type": "baseline", "probe_label": "baseline"})
        rows.append(baseline)

        for label, variant in perturbations(champ):
            r = evaluate(variant, h4_long, h4, m15)
            r.update({"champion_id": cid, "probe_type": "perturbation", "probe_label": label})
            rows.append(r)

        for label, variant in attribution(champ):
            r = evaluate(variant, h4_long, h4, m15)
            r.update({"champion_id": cid, "probe_type": "attribution", "probe_label": label})
            rows.append(r)

        for label, variant in coverage(champ):
            r = evaluate(variant, h4_long, h4, m15)
            r.update({"champion_id": cid, "probe_type": "coverage", "probe_label": label})
            rows.append(r)

        print(f"  {cid}: {len(perturbations(champ))} perturbations, "
              f"{len(attribution(champ))} attributions, "
              f"{len(coverage(champ))} coverage probes")

    out_df = pd.DataFrame(rows)
    cols_first = ["champion_id", "probe_type", "probe_label", "spec_id"]
    cols = cols_first + [c for c in out_df.columns if c not in cols_first]
    out_df = out_df[cols]
    out_df.to_csv(SKEPTIC, index=False)

    print(f"\n=== skeptic complete: {len(out_df)} probes across {len(champions)} champions ===")
    # Quick highlights per champion
    for champ in champions:
        cid = champ["id"]
        sub = out_df[out_df["champion_id"] == cid]
        base = sub[sub["probe_type"] == "baseline"].iloc[0]
        print(f"\n  champion: {cid}  baseline wf_sharpe={base['wf_median_sharpe']}")

        # cost break-even
        cost = sub[sub["probe_label"].str.startswith("cost_bps=")].copy()
        if not cost.empty:
            cost["bps"] = cost["probe_label"].str.split("=").str[1].astype(float)
            broken = cost[~cost["still_certified"]].sort_values("bps")
            survives = cost[cost["still_certified"]].sort_values("bps")
            if not broken.empty:
                first_break = float(broken["bps"].iloc[0])
                print(f"    cost break-even: still certified up to {survives['bps'].max() if not survives.empty else 0} bps; first break at {first_break} bps")

        # attribution drop
        attrib = sub[sub["probe_type"] == "attribution"]
        if not attrib.empty:
            for _, r in attrib.iterrows():
                drop_share = r["wf_median_sharpe"] - base["wf_median_sharpe"]
                print(f"    {r['probe_label']:30s} wf_sharpe={r['wf_median_sharpe']:.3f}  Δ={drop_share:+.3f}  certified={r['still_certified']}")

        # coverage hits
        cov_cert = sub[(sub["probe_type"] == "coverage") & (sub["still_certified"])]
        if not cov_cert.empty:
            print(f"    coverage hits (would-have-certified, off-grid):")
            for _, r in cov_cert.iterrows():
                print(f"      {r['probe_label']:30s} wf_sharpe={r['wf_median_sharpe']:.3f} ho_ret={r['ho_total_return']:+.4f}")

    print(f"\nWrote: {SKEPTIC}")


if __name__ == "__main__":
    run()
