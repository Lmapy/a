"""Regression tests for the stop-exit spread bug.

Bug history:
  1. The old executor charged the bucket-FINAL bar's spread on the exit
     leg even when a stop-loss fired earlier in the bucket. This
     over/under-stated cost whenever spread changed across the H4 bucket.
  2. The old executor multiplied `spread` by `POINT_SIZE = 0.001` even
     though Dukascopy candles store `spread = ask - bid` already in
     price units. That undercharged costs by ~1000x. Fixed in the
     hardening pass: spread per leg is the spread value directly.

These tests synthesise a small (h4, m15) pair where the spread varies
across the bucket and an early stop is forced, then assert that the
trade's `spread_paid` (or equivalent cost) matches entry-bar + exit-bar
spreads in price units, NOT entry-bar + last-bar and NOT scaled by
POINT_SIZE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.types import Spec
from execution.executor import ExecutionModel, run as run_exec


# Realistic XAUUSD spreads in price units ($/oz):
# normal hours ~$0.30, news event ~$0.80.
ENTRY_SPREAD = 0.30
MID_SPREAD = 0.40
EXIT_SPREAD = 0.80    # spike at the bar where the stop will fire
TAIL_SPREAD = 0.20
TOL = 1e-6


def _make_h4_m15_with_varying_spread():
    """One H4 bucket, 16 M15 sub-bars, with the stop hitting at sub-bar #2.

    Spread per sub-bar (price units, $/oz):
        idx 0 (entry):      0.30
        idx 1:              0.40
        idx 2 (exit/stop):  0.80   <- spike
        idx 3..15:          0.20
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
    spreads = [ENTRY_SPREAD, MID_SPREAD, EXIT_SPREAD] + [TAIL_SPREAD] * 13
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
    """v2 executor: exit-leg spread must match the bar where the stop fired,
    and the round-trip cost must be in price units (entry + exit, no
    POINT_SIZE rescale)."""
    h4, m15 = _make_h4_m15_with_varying_spread()

    spec = Spec(
        id="t",
        filters=[],
        entry={"type": "touch_entry"},
        stop={"type": "prev_h4_extreme"},
        exit={"type": "h4_close"},
    )
    em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                        miss_prob_market=0.0, miss_prob_limit=0.0,
                        spread_mult=1.0)
    trades = run_exec(spec, h4, m15, em)
    assert len(trades) == 1, f"expected 1 trade, got {len(trades)}"
    t = trades[0]

    # bug 1 (bucket-final exit spread): would have been ENTRY + TAIL = 0.50
    bug1 = ENTRY_SPREAD + TAIL_SPREAD
    # bug 2 (POINT_SIZE undercharge): would have been (ENTRY + EXIT) * 0.001 = 0.0011
    bug2 = (ENTRY_SPREAD + EXIT_SPREAD) * 0.001

    expected = ENTRY_SPREAD + EXIT_SPREAD  # 1.10
    assert abs(t.spread_paid - expected) < TOL, (
        f"v2 spread_paid wrong: got {t.spread_paid}, expected {expected} "
        f"(price units). bug1 (bucket-final)={bug1}, bug2 (POINT_SIZE)={bug2}"
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
    expected = ENTRY_SPREAD + EXIT_SPREAD  # 1.10
    assert abs(t.cost - expected) < TOL, (
        f"v1 cost wrong: got {t.cost}, expected {expected} (price units)"
    )


def test_spread_unit_from_bid_ask():
    """Lock the convention: cost charged equals (entry ask-bid + exit ask-bid)
    in price units, when the candle's `spread` column was constructed as
    ask - bid (the Dukascopy codec convention)."""
    h4, m15 = _make_h4_m15_with_varying_spread()
    # Re-derive `spread` as if from synthetic bid/ask. For each m15 bar we
    # treat `spread` as ask - bid; close = mid; bid = mid - spread/2; ask = mid + spread/2.
    # Then verify that cost == sum of (ask - bid) on entry and exit bars.
    asks_minus_bids = m15["spread"].values  # already ask - bid by construction
    entry_idx = 0
    exit_idx = 2  # where the stop fires (engineered above)
    expected_cost = float(asks_minus_bids[entry_idx]) + float(asks_minus_bids[exit_idx])

    spec = Spec(
        id="t",
        filters=[],
        entry={"type": "touch_entry"},
        stop={"type": "prev_h4_extreme"},
        exit={"type": "h4_close"},
    )
    em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                        miss_prob_market=0.0, miss_prob_limit=0.0,
                        spread_mult=1.0)
    trades = run_exec(spec, h4, m15, em)
    assert len(trades) == 1
    assert abs(trades[0].spread_paid - expected_cost) < TOL, (
        f"cost convention drift: got {trades[0].spread_paid}, "
        f"expected {expected_cost} = ask-minus-bid at entry + at exit"
    )


def test_spread_mult_scales_cost():
    """Stress mode multiplies the per-bar spread by `spread_mult`.
    Verify the math is straight in price units."""
    h4, m15 = _make_h4_m15_with_varying_spread()
    spec = Spec(
        id="t",
        filters=[],
        entry={"type": "touch_entry"},
        stop={"type": "prev_h4_extreme"},
        exit={"type": "h4_close"},
    )
    base_em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                             miss_prob_market=0.0, miss_prob_limit=0.0,
                             spread_mult=1.0)
    stress_em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                               miss_prob_market=0.0, miss_prob_limit=0.0,
                               spread_mult=2.0)
    base = run_exec(spec, h4, m15, base_em)[0]
    stress = run_exec(spec, h4, m15, stress_em)[0]
    assert abs(stress.spread_paid - 2.0 * base.spread_paid) < TOL, (
        f"spread_mult=2.0 should double cost; "
        f"base={base.spread_paid}, stress={stress.spread_paid}"
    )


if __name__ == "__main__":
    test_stopped_trade_uses_actual_exit_bar_spread_v1()
    print("  PASS  test_stopped_trade_uses_actual_exit_bar_spread_v1")
    test_stopped_trade_uses_actual_exit_bar_spread_v2()
    print("  PASS  test_stopped_trade_uses_actual_exit_bar_spread_v2")
    test_spread_unit_from_bid_ask()
    print("  PASS  test_spread_unit_from_bid_ask")
    test_spread_mult_scales_cost()
    print("  PASS  test_spread_mult_scales_cost")
