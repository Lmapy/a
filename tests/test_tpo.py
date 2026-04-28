"""TPO module tests.

Verify:
  * TPO uses time-at-price only; no volume column required.
  * POC, VAH, VAL are sane on a synthetic profile.
  * Single prints / poor highs / poor lows / excess flags follow
    auction-theory definitions.
  * Initial balance covers the first N brackets only.
  * Open inside / outside prior value works when prior value is
    supplied.
  * Empty input does not crash.
  * Profile JSON round-trips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.tpo import (
    TPOProfile, compute_tpo_profile, session_slices,
)


def _candles_at_prices(price_seq, *, start="2024-01-01 13:00", freq="15min"):
    """Build a candle frame whose high == low == price (degenerate
    wide-zero candles) so we can hand-compute the TPO. Works for
    deterministic tests."""
    times = pd.date_range(start, periods=len(price_seq), freq=freq, tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "open": price_seq, "high": price_seq, "low": price_seq, "close": price_seq,
    })
    return df


def test_no_volume_column_required():
    df = _candles_at_prices([2000.0, 2000.5, 2001.0])
    assert "volume" not in df.columns
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15)
    # Three brackets, each touching exactly one price -> poc somewhere
    assert p.n_brackets == 3
    # All three prices should appear in the count map
    assert set(p.tpo_counts.keys()) == {2000.0, 2000.5, 2001.0}


def test_poc_is_most_visited_price():
    """Build a session where price camps at 2001.0 for several brackets
    and visits 2000.0 / 2002.0 once each. POC must be 2001.0."""
    seq = [2000.0,                          # bracket 0
           2001.0, 2001.0, 2001.0, 2001.0,  # 4 brackets at 2001
           2002.0]                          # bracket 5
    df = _candles_at_prices(seq, freq="15min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15)
    assert abs(p.poc - 2001.0) < 1e-9
    assert p.tpo_counts[2001.0] == 4


def test_vah_val_sandwich_poc():
    seq = [2000.0, 2000.5, 2001.0, 2001.0, 2001.5, 2002.0]
    df = _candles_at_prices(seq, freq="15min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15,
                              value_area_pct=0.70)
    assert p.val <= p.poc <= p.vah
    assert p.value_area_width >= 0.0


def test_single_prints_listed():
    """Prices that appear in exactly one bracket are single prints."""
    seq = [2000.0,                                # touched once
           2001.0, 2001.0, 2001.0,                # touched 3x
           2002.5]                                # touched once
    df = _candles_at_prices(seq, freq="15min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15)
    assert 2000.0 in p.single_prints
    assert 2002.5 in p.single_prints
    assert 2001.0 not in p.single_prints


def test_poor_high_when_top_revisited():
    """If the top price has count > 1 it is a 'poor high' — the edge
    was tested multiple times rather than spiked through with a single
    print. Auction-theory typical re-test setup."""
    seq = [2000.0, 2002.0, 2002.0]   # top=2002 visited twice
    df = _candles_at_prices(seq, freq="15min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15)
    assert p.poor_high is True
    assert p.excess_high is False


def test_excess_high_when_top_is_single_print():
    seq = [2000.0, 2000.5, 2002.5]   # top=2002.5 single print
    df = _candles_at_prices(seq, freq="15min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15)
    assert p.excess_high is True
    assert p.poor_high is False


def test_initial_balance_covers_first_n_brackets():
    # five brackets; ib_brackets = 2 should cover [0,1] only.
    seq = [2000.0, 2001.0, 2002.0, 2003.0, 2004.0]
    df = _candles_at_prices(seq, freq="30min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=30,
                              initial_balance_brackets=2)
    assert p.initial_balance_low == 2000.0
    assert p.initial_balance_high == 2001.0
    # later brackets do NOT bump IB
    assert p.session_high == 2004.0


def test_open_inside_value_with_prior():
    df = _candles_at_prices([2000.5, 2001.0, 2001.5], freq="15min")
    # prior value area = (2000.0, 2002.0). open=2000.5 is INSIDE.
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15,
                              prior_value_area=(2000.0, 2002.0))
    assert p.open_inside_value is True
    assert p.open_outside_value is False


def test_open_outside_value_with_prior():
    df = _candles_at_prices([2010.0, 2010.5, 2011.0], freq="15min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15,
                              prior_value_area=(2000.0, 2002.0))
    assert p.open_outside_value is True


def test_empty_input_safe():
    df = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=30)
    assert p.n_brackets == 0
    assert p.tpo_counts == {}


def test_to_json_serialises_cleanly():
    df = _candles_at_prices([2000.0, 2001.0, 2001.0, 2002.0], freq="15min")
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15)
    payload = p.to_json()
    import json
    s = json.dumps(payload, default=str)
    assert "poc" in s and "vah" in s and "val" in s


def test_session_slices_yields_per_day_groups():
    """Three explicit calendar days of M15 candles spanning the full
    11:00-22:00 UTC range. The slicer with default 13:00-20:00 UTC
    must yield exactly 3 groups, each with 28 bars (7 hours × 4)."""
    rows = []
    for day in range(3):
        base = pd.Timestamp(f"2024-01-0{day+1} 11:00", tz="UTC")
        times = [base + pd.Timedelta(minutes=15 * k) for k in range(44)]
        for t in times:
            rows.append({"time": t, "open": 2000.0, "high": 2001.0,
                         "low": 1999.0, "close": 2000.5})
    df = pd.DataFrame(rows)
    groups = list(session_slices(df))
    assert len(groups) == 3, f"expected 3 day groups, got {len(groups)}"
    for g in groups:
        assert len(g) == 28, f"expected 28 bars per session, got {len(g)}"


def test_bracket_count_uses_candle_range_not_volume():
    """A wide-range candle should populate every bin between its low
    and high — even though we pass no `volume`."""
    times = pd.date_range("2024-01-01 13:00", periods=2, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "open": [2000.0, 2002.0],
        "high": [2001.0, 2002.5],
        "low":  [1999.5, 2001.0],
        "close": [2000.5, 2002.5],
    })
    p = compute_tpo_profile(df, bin_size=0.5, bracket_minutes=15)
    # bracket 0: low 1999.5 -> high 1001.0 covers 1999.5/2000.0/2000.5/2001.0
    # bracket 1: 2001.0 -> 2002.5 covers 2001.0/2001.5/2002.0/2002.5
    # 2001.0 is touched by both brackets
    assert p.tpo_counts.get(2001.0, 0) == 2


if __name__ == "__main__":
    fns = [
        test_no_volume_column_required,
        test_poc_is_most_visited_price,
        test_vah_val_sandwich_poc,
        test_single_prints_listed,
        test_poor_high_when_top_revisited,
        test_excess_high_when_top_is_single_print,
        test_initial_balance_covers_first_n_brackets,
        test_open_inside_value_with_prior,
        test_open_outside_value_with_prior,
        test_empty_input_safe,
        test_to_json_serialises_cleanly,
        test_session_slices_yields_per_day_groups,
        test_bracket_count_uses_candle_range_not_volume,
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
