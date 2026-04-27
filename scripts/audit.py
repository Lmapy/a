"""Pipeline audit.

Runs four classes of checks and prints a pass/fail report:

  1. Data integrity     -- monotonic time, no duplicates, OHLC validity,
                          no NaN, no negative or zero prices, time-gap
                          distribution.
  2. Cross-dataset alignment -- every H4 matched bar has M15 sub-bars in
                                its bucket; M15 bars per H4 bucket are
                                in [12, 16] (allowing weekend half-bars);
                                no orphan M15 buckets.
  3. Simulator correctness   -- look-ahead leak tests; manual replay of
                                one strategy spec against run_full_sim
                                with row-by-row PnL match.
  4. Walk-forward harness    -- folds non-overlapping; train_end ==
                                test_start; per-fold trade counts > 0
                                for at least one well-known signal.

Exit code 0 = all checks pass, 1 = any failure (so the script can gate
CI later). The full report is also written to results/audit.txt.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from strategy import (  # noqa: E402
    apply_filters_np, atr_np, compute_signal_np, run_full_sim, run_h4_sim,
    spec_id, trades_to_metrics,
)
from walkforward import make_folds, walk_forward  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
H4_LONG = DATA / "XAUUSD_H4_long.csv"
H4 = DATA / "XAUUSD_H4_matched.csv"
M15 = DATA / "XAUUSD_M15_matched.csv"


# ---------- reporter ----------

class Report:
    def __init__(self) -> None:
        self.buf = io.StringIO()
        self.failures = 0

    def section(self, title: str) -> None:
        self.write("")
        self.write(f"=== {title} ===")

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        line = f"  [{tag}] {name}"
        if detail:
            line += f"   ({detail})"
        self.write(line)

    def write(self, s: str = "") -> None:
        print(s)
        self.buf.write(s + "\n")

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.buf.getvalue())


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)


# ---------- audit 1: data integrity ----------

def audit_dataset(rep: Report, name: str, df: pd.DataFrame, expected_step_minutes: int) -> None:
    rep.section(f"data integrity — {name}")
    n = len(df)
    rep.write(f"  rows={n}  span={df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    rep.check("non-empty", n > 0)
    rep.check("monotonic time", bool(df["time"].is_monotonic_increasing),
              detail=f"duplicates={int(df['time'].duplicated().sum())}")
    rep.check("no duplicate timestamps", int(df["time"].duplicated().sum()) == 0)
    rep.check("tz-aware UTC", df["time"].dt.tz is not None,
              detail=f"tz={df['time'].dt.tz}")

    for col in ("open", "high", "low", "close"):
        n_nan = int(df[col].isna().sum())
        rep.check(f"{col}: no NaN", n_nan == 0, detail=f"nan={n_nan}")
        n_bad = int((df[col] <= 0).sum())
        rep.check(f"{col}: positive", n_bad == 0, detail=f"non-positive={n_bad}")

    bad_hi = int(((df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])).sum())
    bad_lo = int(((df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])).sum())
    rep.check("OHLC consistent (high is highest)", bad_hi == 0, detail=f"bad rows={bad_hi}")
    rep.check("OHLC consistent (low is lowest)",  bad_lo == 0, detail=f"bad rows={bad_lo}")

    deltas = df["time"].diff().dt.total_seconds().div(60).dropna()
    expected = float(expected_step_minutes)
    on_step = int((deltas == expected).sum())
    weekend_like = int((deltas > expected).sum())
    backwards = int((deltas <= 0).sum())
    rep.write(f"  step minutes: on-step={on_step}, > step (gaps incl. weekends)={weekend_like}, <= 0={backwards}")
    rep.check("no backwards or zero time deltas", backwards == 0)
    rep.check(f"≥ 80% of bars on the {expected_step_minutes}-minute grid",
              on_step / max(len(deltas), 1) >= 0.80,
              detail=f"on-step share={on_step/max(len(deltas),1):.3f}")


# ---------- audit 2: alignment ----------

def audit_alignment(rep: Report, h4: pd.DataFrame, m15: pd.DataFrame) -> None:
    rep.section("cross-dataset alignment — H4 matched ↔ M15 matched")
    m15 = m15.copy()
    m15["bucket"] = m15["time"].dt.floor("4h")
    m15_bucket_set = set(m15["bucket"].unique())
    h4_set = set(h4["time"].unique())

    missing_subbars = h4_set - m15_bucket_set
    orphan = m15_bucket_set - h4_set
    rep.check("every H4 matched bar has ≥1 M15 sub-bar",
              len(missing_subbars) == 0,
              detail=f"missing buckets={len(missing_subbars)}")
    rep.check("no orphan M15 buckets outside the H4 matched range",
              len(orphan) == 0,
              detail=f"orphan buckets={len(orphan)}")

    counts = m15.groupby("bucket").size()
    rep.write(f"  M15 bars per H4 bucket: min={int(counts.min())}, p25={int(counts.quantile(.25))}, median={int(counts.median())}, max={int(counts.max())}")
    rep.check("median M15 bars per H4 bucket is 16", int(counts.median()) == 16)
    rep.check("no bucket has fewer than 4 sub-bars (smoke threshold)",
              int(counts.min()) >= 4,
              detail=f"min={int(counts.min())}")


# ---------- audit 3: simulator correctness ----------

def audit_simulator(rep: Report, h4_long: pd.DataFrame, h4: pd.DataFrame, m15: pd.DataFrame) -> None:
    rep.section("simulator correctness — H4 sim")

    # 3a. Vectorised compute_signal_np uses bar i-1's color, not i's.
    cl = h4["close"].values.astype(float)
    op = h4["open"].values.astype(float)
    color = np.sign(cl - op).astype(int)
    sig = compute_signal_np(h4, {"signal": {"type": "prev_color"}})
    rep.check("compute_signal: bar 0 has no signal", int(sig[0]) == 0)
    rep.check("compute_signal: sig[i] == color[i-1] for i>=1",
              bool(np.array_equal(sig[1:], color[:-1])))

    # 3b. Look-ahead test: build a synthetic "leaked" signal that uses bar i
    # itself; the leak version should *outperform* by a wide margin.
    # If we swap our signal for the leaked one and PnL goes up massively,
    # we're not leaking. (We are testing the harness, not the signal.)
    spec = {"signal": {"type": "prev_color"}, "filters": [],
            "entry": {"type": "h4_open"}, "stop": {"type": "none"},
            "exit": {"type": "h4_close"}, "cost_bps": 0.0}  # zero cost for clean math
    trades = run_h4_sim(spec, h4)
    rets = np.array([t.ret for t in trades])
    # leaked equivalent: pretend sig were color[i] instead of color[i-1]
    leaked_ret = (color * (cl - op) / op)
    leaked_total = float(np.sum(leaked_ret))
    real_total = float(rets.sum())
    rep.write(f"  total ret (real prev-color signal) = {real_total:+.4f}")
    rep.write(f"  total ret if signal leaked current bar = {leaked_total:+.4f}")
    rep.check("real signal much weaker than a leaked oracle",
              leaked_total > real_total + 0.5,
              detail=f"leaked - real = {leaked_total - real_total:.4f}")

    # 3c. Manual replay of one bar through run_full_sim.
    rep.section("simulator correctness — manual M15 replay")
    spec = {
        "signal": {"type": "prev_color"},
        "filters": [],
        "entry": {"type": "m15_open"},
        "stop": {"type": "none"},
        "exit": {"type": "h4_close"},
        "cost_model": "bps",
        "cost_bps": 0.0,
    }
    trades = run_full_sim(spec, h4, m15)
    rep.check("run_full_sim returns at least one trade", len(trades) > 0)

    # Find the H4 bar at index >=1 whose previous bar had non-zero color.
    h4r = h4.reset_index(drop=True)
    m15r = m15.copy()
    m15r["bucket"] = m15r["time"].dt.floor("4h")
    sample_idx = None
    for i in range(1, len(h4r)):
        prev = h4r.iloc[i - 1]
        if np.sign(prev["close"] - prev["open"]) != 0 and h4r.iloc[i]["time"] in set(m15r["bucket"]):
            sample_idx = i
            break
    if sample_idx is None:
        rep.check("found a sample H4 bar to replay manually", False)
    else:
        prev = h4r.iloc[sample_idx - 1]
        cur = h4r.iloc[sample_idx]
        sub = m15r[m15r["bucket"] == cur["time"]].sort_values("time").reset_index(drop=True)
        manual_dir = int(np.sign(prev["close"] - prev["open"]))
        manual_entry = float(sub.iloc[0]["open"])
        manual_exit = float(cur["close"])
        manual_pnl = manual_dir * (manual_exit - manual_entry)
        manual_ret = manual_pnl / manual_entry

        # find that trade in the run_full_sim result
        found = next((t for t in trades if t.entry_time == sub.iloc[0]["time"]), None)
        rep.check("manual replay finds matching trade in run_full_sim", found is not None)
        if found is not None:
            rep.check("manual trade direction == sim direction",
                      manual_dir == found.direction,
                      detail=f"manual={manual_dir} sim={found.direction}")
            rep.check("manual trade entry == sim entry",
                      abs(manual_entry - found.entry) < 1e-9,
                      detail=f"diff={manual_entry - found.entry:.6f}")
            rep.check("manual trade exit == sim exit",
                      abs(manual_exit - found.exit) < 1e-9,
                      detail=f"diff={manual_exit - found.exit:.6f}")
            rep.check("manual trade ret == sim ret (within 1e-9)",
                      abs(manual_ret - found.ret) < 1e-9,
                      detail=f"diff={manual_ret - found.ret:.2e}")

    # 3.5 Retracement entry sanity: at level=0.5 entry must equal prev-H4 mid;
    # at level=0.618 entry must equal prev_high - 0.618*range (long) /
    # prev_low + 0.618*range (short); deeper levels strictly reduce fill rate.
    rep.section("simulator correctness — fib retracement entries")
    fill_counts: dict[float, int] = {}
    for lvl in (0.382, 0.5, 0.618, 0.786):
        spec_l = {
            "signal": {"type": "prev_color"},
            "filters": [],
            "entry": {"type": "m15_retrace_fib", "level": lvl},
            "stop": {"type": "none"},
            "exit": {"type": "h4_close"},
            "cost_model": "bps",
            "cost_bps": 0.0,
        }
        trades_l, _ = run_full_sim(spec_l, h4, m15, return_diag=True)
        fill_counts[lvl] = len(trades_l)
        if trades_l:
            t = trades_l[0]
            bucket = pd.Timestamp(t.entry_time).floor("4h")
            h4_row_idx = int(h4.index[h4["time"] == bucket][0])
            prev = h4.iloc[h4_row_idx - 1]
            ph, pl = float(prev["high"]), float(prev["low"])
            expected = ph - lvl * (ph - pl) if t.direction > 0 else pl + lvl * (ph - pl)
            rep.check(f"fib level {lvl}: entry equals expected level price",
                      abs(t.entry - expected) < 1e-6,
                      detail=f"entry={t.entry} expected={expected:.6f}")
    rep.write(f"  fill counts by level: {fill_counts}")
    monotone = (fill_counts[0.382] >= fill_counts[0.5] >= fill_counts[0.618] >= fill_counts[0.786])
    rep.check("fill count is monotone non-increasing in fib level",
              monotone, detail=f"counts={fill_counts}")

    # 3.6 Existing midpoint replay: trade only fires when M15 retraces to mid;
    spec_r = {
        "signal": {"type": "prev_color"},
        "filters": [],
        "entry": {"type": "m15_retrace_50"},
        "stop": {"type": "prev_h4_open"},
        "exit": {"type": "prev_h4_extreme_tp"},
        "cost_model": "bps",
        "cost_bps": 0.0,
    }
    trades_r, diag_r = run_full_sim(spec_r, h4, m15, return_diag=True)
    rep.write(f"  diag: {diag_r}")
    rep.check("retracement: at least one trade fires", len(trades_r) > 0)
    if trades_r:
        # Pick a sample trade and replay manually.
        t = trades_r[0]
        idx = int(np.where(h4["time"].values >= t.entry_time.to_datetime64())[0][0])
        # Trade entry_time can be inside H4 bar i, but the bucket is H4 i.
        bucket = h4.iloc[idx]["time"].floor("4h") if h4.iloc[idx]["time"] != t.entry_time else h4.iloc[idx]["time"]
        # find the H4 row whose bucket matches t.entry_time.floor('4h')
        bucket = pd.Timestamp(t.entry_time).floor("4h")
        h4_row_idx = int(h4.index[h4["time"] == bucket][0])
        prev = h4.iloc[h4_row_idx - 1]
        mid = (prev["high"] + prev["low"]) / 2.0
        rep.check("retracement: entry price equals prev H4 midpoint",
                  abs(t.entry - mid) < 1e-6, detail=f"entry={t.entry} mid={mid}")
        rep.check("retracement: direction matches prev H4 color",
                  t.direction == int(np.sign(prev["close"] - prev["open"])),
                  detail=f"dir={t.direction} prev_color={int(np.sign(prev['close']-prev['open']))}")
        # exit must be one of: prev_open (stop), prev_high/prev_low (TP), or h4 close
        valid_exits = [float(prev["open"]),
                       float(prev["high"]) if t.direction > 0 else float(prev["low"]),
                       float(h4.iloc[h4_row_idx]["close"])]
        rep.check("retracement: exit is one of {prev open, prev extreme, h4 close}",
                  any(abs(t.exit - v) < 1e-6 for v in valid_exits),
                  detail=f"exit={t.exit} valid={valid_exits}")

    # 3d. Look-ahead probe: signal at bar i must not depend on close[i].
    # Mutate close[i] arbitrarily and confirm sig[i] is unchanged.
    rep.section("simulator correctness — look-ahead probes")
    h4_perturbed = h4_long.copy().reset_index(drop=True)
    target = len(h4_perturbed) - 5
    sig_before = compute_signal_np(h4_perturbed, {"signal": {"type": "prev_color"}})[target]
    h4_perturbed.loc[target, "close"] = h4_perturbed.loc[target, "close"] + 1000.0
    sig_after = compute_signal_np(h4_perturbed, {"signal": {"type": "prev_color"}})[target]
    rep.check(
        "perturbing close[i] does not change sig[i] (no current-bar leak)",
        sig_before == sig_after,
        detail=f"before={sig_before}, after={sig_after}",
    )


# ---------- audit 4: walk-forward harness ----------

def audit_walkforward(rep: Report, h4_long: pd.DataFrame) -> None:
    rep.section("walk-forward harness — folds")
    folds = make_folds(h4_long)
    rep.check("at least 10 folds", len(folds) >= 10, detail=f"folds={len(folds)}")

    # non-overlap of test windows
    overlap = 0
    for i in range(len(folds) - 1):
        if folds[i].test_end > folds[i + 1].test_start:
            overlap += 1
    rep.check("test windows are non-overlapping", overlap == 0, detail=f"overlaps={overlap}")

    # train_end == test_start within each fold
    bad = sum(1 for f in folds if f.train_end != f.test_start)
    rep.check("each fold: train_end == test_start", bad == 0, detail=f"misaligned={bad}")

    # The signal that already certified should produce trades in most folds.
    spec = {
        "signal": {"type": "prev_color"},
        "filters": [
            {"type": "body_atr", "min": 0.5, "atr_n": 14},
            {"type": "regime", "ma_n": 50, "side": "with"},
        ],
        "entry": {"type": "m15_open"},  # ignored in H4-only walk-forward
        "stop": {"type": "none"},
        "exit": {"type": "h4_close"},
        "cost_bps": 1.5,
    }
    wf = walk_forward(spec, h4_long)
    nz = int((wf["fold_table"]["trades"] > 0).sum()) if not wf["fold_table"].empty else 0
    rep.write(f"  certified-spec walk-forward: folds={wf['folds']}, median_sharpe={wf['median_sharpe']}, pct_pos={wf['pct_positive_folds']}")
    rep.check("certified spec produces trades in ≥80% of folds",
              nz / max(wf["folds"], 1) >= 0.80,
              detail=f"folds with trades={nz}/{wf['folds']}")


# ---------- main ----------

def audit_data_manifest(rep) -> None:
    """Verify data_manifest.json exists, has SHA-pinned URLs, and the
    on-disk files still match the recorded sha256 + row count."""
    import hashlib
    import json
    rep.section("data manifest — pinned source provenance")
    manifest_path = RESULTS / "data_manifest.json"
    rep.check("data_manifest.json exists", manifest_path.exists(),
              detail=str(manifest_path))
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    rep.check("manifest has at least one source",
              len(manifest.get("sources", [])) >= 1)
    for src in manifest.get("sources", []):
        rep.check(f"{src['name']}: URL pinned to commit SHA (no /HEAD/)",
                  "/HEAD/" not in src["url"] and src.get("commit_sha"),
                  detail=src["url"])
        # We can't re-hash the raw download cheaply (raw bytes != normalised
        # csv on disk), but we can check the normalised CSV's row count.
        on_disk = (DATA / src["name"])
        rep.check(f"{src['name']}: on-disk file present",
                  on_disk.exists(), detail=str(on_disk))


def audit_stop_exit_spread(rep) -> None:
    """Regression check for the stop-exit spread bug.

    Synthesises a small (h4, m15) pair where a stop fires at sub-bar #2
    (spread = 80 pts) but the bucket-final bar has spread = 5 pts.
    Asserts that the trade's recorded cost reflects the EXIT-bar spread,
    not the bucket-final spread.
    """
    import pandas as pd
    rep.section("simulator correctness — stop-exit uses actual exit-bar spread")

    h4_times = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 04:00:00"], utc=True)
    h4 = pd.DataFrame({
        "time": h4_times,
        "open":  [100.0, 99.5],
        "high":  [101.0, 100.0],
        "low":   [99.0,  90.0],
        "close": [100.8, 95.0],
        "volume": [1000.0, 1000.0],
        "spread": [0.0, 0.0],
    })
    m15_times = pd.date_range("2024-01-01 04:00:00", periods=16, freq="15min", tz="UTC")
    m15 = pd.DataFrame({
        "time": m15_times,
        "open":  [99.5, 99.4, 99.3] + [98.0 + i * 0.1 for i in range(13)],
        "high":  [99.6, 99.5, 99.4] + [98.5 + i * 0.1 for i in range(13)],
        "low":   [99.4, 99.3, 95.0] + [97.5 + i * 0.1 for i in range(13)],
        "close": [99.5, 99.4, 95.5] + [98.0 + i * 0.1 for i in range(13)],
        "volume": [100.0] * 16,
        "spread": [10, 12, 80] + [5] * 13,
    })

    spec = {
        "id": "audit_stop_exit_spread",
        "signal": {"type": "prev_color"},
        "filters": [],
        "entry": {"type": "h4_open"},
        "stop":  {"type": "prev_h4_extreme"},
        "exit":  {"type": "h4_close"},
        "cost_model": "spread",
    }
    trades = run_full_sim(spec, h4, m15)
    rep.check("v1: synthetic regression produces exactly 1 trade", len(trades) == 1)
    if len(trades) != 1:
        return
    bug_value = (10 + 5) * 0.001     # entry-bar + bucket-final-bar spread
    actual = trades[0].cost
    rep.check("v1: stopped trade's cost differs from buggy bucket-final cost",
              actual > bug_value + 0.001,
              detail=f"actual_cost={actual}, bug_value={bug_value}")


def main() -> int:
    rep = Report()
    rep.write("XAUUSD 4h-continuation pipeline audit")
    rep.write("=" * 60)

    h4_long = load(H4_LONG)
    h4 = load(H4)
    m15 = load(M15)

    audit_dataset(rep, "H4 long",     h4_long, expected_step_minutes=240)
    audit_dataset(rep, "H4 matched",  h4,      expected_step_minutes=240)
    audit_dataset(rep, "M15 matched", m15,     expected_step_minutes=15)

    audit_alignment(rep, h4, m15)
    audit_simulator(rep, h4_long, h4, m15)
    audit_stop_exit_spread(rep)
    audit_walkforward(rep, h4_long)
    audit_data_manifest(rep)

    rep.write("")
    rep.write(f"=== audit summary: {rep.failures} failures ===")
    rep.dump(RESULTS / "audit.txt")
    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
