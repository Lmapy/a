"""Power tests for the new statistical probes.

For each test we want both halves of the power story:

  - rejects an obvious-edge synthetic strategy (low p-value)
  - does NOT reject a random-direction strategy with same trade timing
    (p-value distributed roughly uniformly, definitely not consistently
    below 0.05)

Plus: the deprecated `shuffled_outcome_test` must raise on any call.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.types import Trade
from validation.statistical_tests import (
    benjamini_hochberg,
    daily_block_bootstrap_test,
    label_permutation_test,
    random_eligible_entry_test,
    shuffled_outcome_test,
)


def _trades_from_returns(rets: np.ndarray, start_date: str = "2024-01-01") -> list[Trade]:
    times = pd.date_range(start_date, periods=len(rets), freq="4h", tz="UTC")
    out = []
    for r, t in zip(rets, times):
        out.append(Trade(
            entry_time=t, exit_time=t,
            direction=1 if r >= 0 else -1,
            entry=2000.0,
            exit=2000.0 * (1.0 + float(r)),
            cost=0.0, pnl=float(r) * 2000.0, ret=float(r),
        ))
    return out


def _h4_synth(n: int = 300, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rets = rng.normal(0, 0.005, n)
    closes = 2000.0 * np.cumprod(1.0 + rets)
    opens = np.concatenate([[2000.0], closes[:-1]])
    return pd.DataFrame({
        "time": times, "open": opens, "high": closes * 1.002,
        "low": closes * 0.998, "close": closes,
        "volume": np.full(n, 1000.0), "spread": np.full(n, 0.30),
    })


# ---------- shuffle test must be dead ----------

def test_shuffled_outcome_test_is_deprecated():
    """Calling the old test must raise. It was a no-op (Sharpe is
    invariant to per-trade permutation)."""
    rets = np.random.default_rng(0).normal(0.001, 0.01, 50)
    trades = _trades_from_returns(rets)
    raised = False
    try:
        shuffled_outcome_test(trades, n_perm=10)
    except RuntimeError as exc:
        raised = "deprecated" in str(exc).lower()
    assert raised, "shuffled_outcome_test must raise RuntimeError on call"


# ---------- label permutation ----------

def test_label_permutation_rejects_strong_edge():
    """A strategy with a strong directional edge must be rejected
    against the label-permutation null."""
    # Strongly positive returns, all signed +1.
    rng = np.random.default_rng(3)
    rets = np.abs(rng.normal(0, 0.01, 80)) + 0.01  # all positive
    trades = _trades_from_returns(rets)
    res = label_permutation_test(trades, n_perm=500, seed=3)
    assert res["passes"] is True, (
        f"strong edge should pass (p < 0.05), got {res}")
    assert res["p_value"] < 0.05


def test_label_permutation_does_not_falsely_pass_noise():
    """A noise strategy (zero-mean returns, random signs) must NOT
    consistently pass at p<0.05 across many seeds."""
    fail_rate = 0
    n_runs = 30
    for seed in range(n_runs):
        rng = np.random.default_rng(seed + 100)
        # signed_rets has zero mean
        rets = rng.normal(0, 0.01, 80) * rng.choice([-1.0, 1.0], size=80)
        trades = _trades_from_returns(rets)
        res = label_permutation_test(trades, n_perm=200, seed=seed + 200)
        if res["passes"]:
            fail_rate += 1
    # Under H0, type-I rate should be near 0.05; allow a generous bound.
    assert fail_rate / n_runs <= 0.20, (
        f"noise rejected {fail_rate}/{n_runs} times — type-I rate too high")


# ---------- random eligible entry ----------

def test_random_eligible_entry_rejects_strong_edge():
    """When the real strategy clearly outperforms random entries on the
    same eligible bars, the test must mark passes=True."""
    h4 = _h4_synth(300, seed=21)
    bar_ret = (h4["close"].values - h4["open"].values) / h4["open"].values
    # Engineer trades that pick the top 30 bars by absolute return,
    # signed correctly.
    top = np.argsort(np.abs(bar_ret))[-30:]
    rets = bar_ret[top] * np.sign(bar_ret[top])  # all positive
    times = h4["time"].values[top]
    trades = []
    for t, r in zip(times, rets):
        trades.append(Trade(
            entry_time=pd.Timestamp(t), exit_time=pd.Timestamp(t),
            direction=1, entry=2000.0,
            exit=2000.0 * (1.0 + float(r)),
            cost=0.0, pnl=float(r) * 2000.0, ret=float(r),
        ))
    res = random_eligible_entry_test(trades, h4, n_runs=400, seed=21)
    assert res["passes"] is True, f"engineered strong edge should pass: {res}"


def test_random_eligible_entry_does_not_falsely_pass_random():
    """A random-entry strategy on the same h4 must not consistently
    reject the random-eligible-entry null."""
    h4 = _h4_synth(300, seed=22)
    bar_ret = (h4["close"].values - h4["open"].values) / h4["open"].values
    fail_rate = 0
    n_runs = 25
    for seed in range(n_runs):
        rng = np.random.default_rng(seed + 300)
        idxs = rng.choice(len(h4), size=30, replace=False)
        dirs = rng.choice([-1, 1], size=30)
        rets = dirs * bar_ret[idxs]
        times = h4["time"].values[idxs]
        trades = [Trade(entry_time=pd.Timestamp(t), exit_time=pd.Timestamp(t),
                        direction=int(d), entry=2000.0,
                        exit=2000.0 * (1.0 + float(r)),
                        cost=0.0, pnl=float(r) * 2000.0, ret=float(r))
                  for t, d, r in zip(times, dirs, rets)]
        res = random_eligible_entry_test(trades, h4, n_runs=300, seed=seed + 400)
        if res["passes"]:
            fail_rate += 1
    assert fail_rate / n_runs <= 0.20, (
        f"random rejected {fail_rate}/{n_runs} times — type-I too high")


# ---------- daily block bootstrap ----------

def test_block_bootstrap_rejects_strong_edge():
    """A strategy with consistent positive daily returns must be
    rejected against the bootstrap null."""
    # 60 trading days, each contributing ~+0.5%.
    rng = np.random.default_rng(31)
    daily = rng.normal(0.005, 0.001, 60)
    trades = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for i, r in enumerate(daily):
        t = base + pd.Timedelta(days=i)
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1,
            entry=2000.0, exit=2000.0 * (1.0 + float(r)),
            cost=0.0, pnl=float(r) * 2000.0, ret=float(r),
        ))
    res = daily_block_bootstrap_test(trades, n_runs=500, block_days=5, seed=31)
    assert res["passes"] is True, f"clear daily edge should pass: {res}"


def test_block_bootstrap_does_not_falsely_pass_random_days():
    fail_rate = 0
    n_runs = 25
    for seed in range(n_runs):
        rng = np.random.default_rng(seed + 500)
        daily = rng.normal(0.0, 0.005, 60)
        trades = []
        base = pd.Timestamp("2024-01-01", tz="UTC")
        for i, r in enumerate(daily):
            t = base + pd.Timedelta(days=i)
            trades.append(Trade(
                entry_time=t, exit_time=t, direction=1,
                entry=2000.0, exit=2000.0 * (1.0 + float(r)),
                cost=0.0, pnl=float(r) * 2000.0, ret=float(r),
            ))
        res = daily_block_bootstrap_test(trades, n_runs=200, seed=seed + 600)
        if res["passes"]:
            fail_rate += 1
    assert fail_rate / n_runs <= 0.20, (
        f"random rejected {fail_rate}/{n_runs} — type-I too high")


def test_benjamini_hochberg_basic():
    # Mix of small and large p-values; small ones should be kept.
    p = [0.001, 0.01, 0.04, 0.20, 0.50, 0.90]
    keep = benjamini_hochberg(p, q=0.10)
    assert keep[0] is True
    assert keep[-1] is False
    assert sum(keep) >= 1


if __name__ == "__main__":
    fns = [
        test_shuffled_outcome_test_is_deprecated,
        test_label_permutation_rejects_strong_edge,
        test_label_permutation_does_not_falsely_pass_noise,
        test_random_eligible_entry_rejects_strong_edge,
        test_random_eligible_entry_does_not_falsely_pass_random,
        test_block_bootstrap_rejects_strong_edge,
        test_block_bootstrap_does_not_falsely_pass_random_days,
        test_benjamini_hochberg_basic,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    if failures:
        raise SystemExit(1)
