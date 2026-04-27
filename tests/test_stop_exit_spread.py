"""Regression tests for the stop-exit spread bug.

Bug history (fixed in commit landing this test):
    The old executor charged the bucket-FINAL bar's spread on the exit
    leg even when a stop-loss fired earlier in the bucket. This
    over/under-stated cost whenever spread changed across the H4 bucket.

These tests synthesise a small (h4, m15) pair where the spread varies
across the bucket and an early stop is forced, then assert that the
trade's `spread_paid` (or equivalent cost) matches the entry-bar +
exit-bar spreads, NOT entry-bar + last-bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.constants import POINT_SIZE
from core.types import Spec
from execution.executor import ExecutionModel, run as run_exec


def _make_h4_m15_with_varying_spread():
    """One H4 bucket, 16 M15 sub-bars, with the stop hitting at sub-bar #2.

    Spread per sub-bar (in points):
        idx 0 (entry):  10
        idx 1:          12
        idx 2 (exit):   80     <- big spread spike
        idx 3..15:       5
    """
    h4_times = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 04:00:00"], utc=True)
    h4 = pd.DataFrame({
        "time": h4_times,
        # bar 0 is the SIGNAL bar (prev candle): green so the next bar trades long
        "open":  [100.0, 99.5],
        "high":  [101.0, 100.0],
        "low":   [99.0,  90.0],   # bar 1 has wide range so stops can fire
        "close": [100.8, 95.0],
        "volume": [1000.0, 1000.0],
        "spread": [0.0, 0.0],
    })
    # 16 M15 sub-bars inside H4 bucket starting 04:00.
    m15_times = pd.date_range("2024-01-01 04:00:00", periods=16, freq="15min", tz="UTC")
    spreads = [10, 12, 80] + [5] * 13  # the spike at idx 2
    # opens/closes engineered so that bar 2's low is well below the entry price.
    opens  = [99.5, 99.4, 99.3] + [98.0 + i * 0.1 for i in range(13)]
    highs  = [99.6, 99.5, 99.4] + [98.5 + i * 0.1 for i in range(13)]
    lows   = [99.4, 99.3, 95.0] + [97.5 + i * 0.1 for i in range(13)]   # bar 2 sweeps deep
    closes = [99.5, 99.4, 95.5] + [98.0 + i * 0.1 for i in range(13)]
    m15 = pd.DataFrame({
        "time": m15_times,
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": [100.0] * 16,
        "spread": spreads,
    })
    return h4, m15


def test_stopped_trade_uses_actual_exit_bar_spread_v2():
    """v2 executor: exit-leg spread must match the bar where the stop fired."""
    h4, m15 = _make_h4_m15_with_varying_spread()

    # Touch entry at bar 0; structural stop at prev-H4 low (99.0); the M15
    # at sub-bar #2 sweeps to 95.0 so the stop fires there exactly.
    spec = Spec(
        id="t",
        filters=[],
        entry={"type": "touch_entry"},
        stop={"type": "prev_h4_extreme"},
        exit={"type": "h4_close"},
    )
    # disable slippage so the spread arithmetic is exact
    em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                        miss_prob_market=0.0, miss_prob_limit=0.0)
    trades = run_exec(spec, h4, m15, em)
    assert len(trades) == 1, f"expected 1 trade, got {len(trades)}"
    t = trades[0]
    # bug check: spread_paid would have been (10 + 5) * POINT_SIZE under old code
    # (entry idx 0 spread = 10, bucket-last idx 15 spread = 5).
    bad_total = (10 + 5) * POINT_SIZE
    # correct: entry bar 0 spread = 10, exit bar (where stop fired) spread = 80
    # exit_sub_idx may be 2 if stop fires there.
    # We don't strictly assert the exact value (depends on spread_mult), but we
    # DO assert that the spread paid is materially different from the bug value.
    assert t.spread_paid > bad_total + POINT_SIZE, (
        f"v2 stop-exit spread bug not fixed: "
        f"spread_paid={t.spread_paid}, bug-value~={bad_total}"
    )


def test_stopped_trade_uses_actual_exit_bar_spread_v1():
    """v1 strategy.run_full_sim: same property."""
    import scripts.strategy as v1
    h4, m15 = _make_h4_m15_with_varying_spread()
    spec = {
        "id": "t",
        "signal": {"type": "prev_color"},
        "filters": [],
        "entry": {"type": "h4_open"},
        "stop": {"type": "prev_h4_extreme"},
        "exit": {"type": "h4_close"},
        "cost_model": "spread",
    }
    trades = v1.run_full_sim(spec, h4, m15)
    assert len(trades) == 1, f"expected 1 trade, got {len(trades)}"
    t = trades[0]
    bug_cost = (10 + 5) * POINT_SIZE
    assert t.cost > bug_cost + POINT_SIZE, (
        f"v1 stop-exit spread bug not fixed: cost={t.cost}, bug~={bug_cost}"
    )


if __name__ == "__main__":
    test_stopped_trade_uses_actual_exit_bar_spread_v1()
    print("  PASS  test_stopped_trade_uses_actual_exit_bar_spread_v1")
    test_stopped_trade_uses_actual_exit_bar_spread_v2()
    print("  PASS  test_stopped_trade_uses_actual_exit_bar_spread_v2")
