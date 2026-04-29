"""No-lookahead invariants for filters and entry decisions.

Principle: at bar H4[i] the entry decision must depend ONLY on bars
H4[0..i-1] (already-completed bars). Mutating bar H4[i] or any later
bar's close/high/low must NOT change the entry decision.

This test runs the executor on synthetic data, captures the entry-bar
indices, mutates the future of the H4 frame at and beyond each entry,
and asserts the trade list is identical. It then mutates a prior bar
to show the filter CAN respond to past changes.

Filters covered (every filter that consumes price-level data on the
current or future bar):
  - atr_percentile
  - pdh_pdl
  - vwap_dist
  - htf_vwap_dist
  - body_atr (sanity)
  - regime, regime_class (sanity)
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.types import Spec
from execution.executor import ExecutionModel, run as run_exec


def _make_synthetic(n_h4: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesise n_h4 H4 bars + 16 M15 sub-bars per H4 bucket.

    OHLC is a random-walk wrapping around 2000.0 with realistic ranges.
    Spread is small and constant.
    """
    rng = np.random.default_rng(seed)
    h4_times = pd.date_range("2024-01-01", periods=n_h4, freq="4h", tz="UTC")
    rets = rng.normal(0, 0.0035, n_h4)
    closes = 2000.0 * np.cumprod(1.0 + rets)
    opens = np.concatenate([[2000.0], closes[:-1]])
    range_pct = np.abs(rng.normal(0.005, 0.002, n_h4))
    highs = np.maximum(opens, closes) * (1.0 + range_pct / 2.0)
    lows = np.minimum(opens, closes) * (1.0 - range_pct / 2.0)
    h4 = pd.DataFrame({
        "time": h4_times,
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": rng.uniform(800, 1200, n_h4),
        "spread": np.full(n_h4, 0.0),
    })

    m15_rows = []
    for i, bucket_t in enumerate(h4_times):
        sub_times = pd.date_range(bucket_t, periods=16, freq="15min", tz="UTC")
        # Simple linear path open->close with small noise on the way.
        path = np.linspace(opens[i], closes[i], 16) + rng.normal(0, range_pct[i] * opens[i] * 0.1, 16)
        sub_h = path + np.abs(rng.normal(0, range_pct[i] * opens[i] * 0.05, 16))
        sub_l = path - np.abs(rng.normal(0, range_pct[i] * opens[i] * 0.05, 16))
        for k in range(16):
            m15_rows.append({
                "time": sub_times[k],
                "open": float(path[k]),
                "high": float(sub_h[k]),
                "low": float(sub_l[k]),
                "close": float(path[k]),
                "volume": 100.0,
                "spread": 0.30,
            })
    m15 = pd.DataFrame(m15_rows)
    return h4, m15


def _trade_keys(trades: list) -> list[tuple]:
    """Stable identity for comparing trade lists (entry_time + direction)."""
    return [(t.entry_time, int(t.direction)) for t in trades]


def _run(spec: Spec, h4: pd.DataFrame, m15: pd.DataFrame) -> list:
    em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                        miss_prob_market=0.0, miss_prob_limit=0.0,
                        spread_mult=1.0)
    return run_exec(spec, h4, m15, em)


def _mutate_future(h4: pd.DataFrame, from_idx: int, rng: np.random.Generator) -> pd.DataFrame:
    """Replace H4 close/high/low for bars >= from_idx with extreme values.
    The entry decision at H4[from_idx] must NOT be sensitive to these,
    because the bar hasn't closed when the decision is made."""
    h4_m = h4.copy()
    n = len(h4_m)
    # Big random shifts; flip half the bars upward, half downward.
    shifts = rng.choice([-50.0, 50.0], size=n - from_idx)
    h4_m.loc[from_idx:, "close"] = h4_m.loc[from_idx:, "close"].values + shifts
    h4_m.loc[from_idx:, "high"] = np.maximum(h4_m.loc[from_idx:, "high"].values + shifts,
                                             h4_m.loc[from_idx:, "close"].values + 1.0)
    h4_m.loc[from_idx:, "low"] = np.minimum(h4_m.loc[from_idx:, "low"].values + shifts,
                                            h4_m.loc[from_idx:, "close"].values - 1.0)
    return h4_m


def _check_filter_invariant(filter_dict: dict, label: str,
                             entry: dict | None = None,
                             stop: dict | None = None,
                             min_trades: int = 5) -> None:
    rng = np.random.default_rng(42)
    h4, m15 = _make_synthetic()
    spec = Spec(
        id=f"t_{label}",
        signal={"type": "prev_color"},
        filters=[filter_dict],
        entry=entry or {"type": "touch_entry"},
        stop=stop or {"type": "none"},
        exit={"type": "h4_close"},
    )
    base_trades = _run(spec, h4, m15)
    base_keys = _trade_keys(base_trades)

    if len(base_keys) < min_trades:
        # Filter is very restrictive on this fixture. Still verify the
        # mutation invariant: even with 0 trades, mutating the future at
        # arbitrary indices must keep the trade list unchanged.
        for i in range(20, len(h4), 50):
            h4_m = _mutate_future(h4, i, rng)
            new_trades = _run(spec, h4_m, m15)
            assert _trade_keys(new_trades) == base_keys, (
                f"{label}: low-trade-count case still leaks lookahead at idx {i}")
        return

    # For each base entry, mutate the future and re-run.
    for tr in base_trades[:8]:  # cap at 8 entries to keep test fast
        match = h4.index[h4["time"] == tr.h4_bucket]
        if len(match) == 0:
            continue
        i = int(match[0])
        h4_m = _mutate_future(h4, i, rng)
        new_trades = _run(spec, h4_m, m15)
        new_keys = _trade_keys(new_trades)
        pre = [k for k in base_keys if k[0] <= tr.h4_bucket]
        new_pre = [k for k in new_keys if k[0] <= tr.h4_bucket]
        assert pre == new_pre, (
            f"{label}: entries at or before {tr.h4_bucket} changed when "
            f"future bars were mutated\n"
            f"  before: {pre}\n  after:  {new_pre}")


def test_atr_percentile_no_lookahead():
    _check_filter_invariant(
        {"type": "atr_percentile", "window": 50, "atr_n": 14, "lo": 0.2, "hi": 0.8},
        "atr_percentile")


def test_pdh_pdl_no_lookahead_inside():
    _check_filter_invariant(
        {"type": "pdh_pdl", "mode": "inside"},
        "pdh_pdl_inside")


def test_pdh_pdl_no_lookahead_breakout():
    _check_filter_invariant(
        {"type": "pdh_pdl", "mode": "breakout"},
        "pdh_pdl_breakout")


def test_atr_distance_from_session_mean_no_lookahead():
    """Replaces the deleted vwap_dist test. The OHLC-only proxy
    must also be insensitive to future bars."""
    _check_filter_invariant(
        {"type": "atr_distance_from_session_mean", "max_z": 3.0,
         "atr_n": 14, "session_start_hour_utc": 13},
        "atr_distance_from_session_mean")


def test_tpo_value_acceptance_no_lookahead():
    """TPO filter pulls levels from the PREVIOUS session, so it
    cannot reach into the future. The mutation invariant should
    therefore hold."""
    rng = np.random.default_rng(42)
    h4, m15 = _make_synthetic()
    # Attach prev-session TPO columns so the TPO filter has data.
    from analytics.tpo_levels import attach_prev_session_tpo
    h4 = attach_prev_session_tpo(h4, m15)
    spec = Spec(
        id="t_tpo",
        signal={"type": "prev_color"},
        filters=[{"type": "tpo_value_acceptance"}],
        entry={"type": "touch_entry"},
        stop={"type": "none"},
        exit={"type": "h4_close"},
    )
    base_trades = _run(spec, h4, m15)
    base_keys = _trade_keys(base_trades)
    # mutate the future of every other bar; TPO filter should
    # remain stable because it reads PRIOR session data only
    for i in range(20, len(h4), 30):
        h4_m = _mutate_future(h4, i, rng)
        new_trades = _run(spec, h4_m, m15)
        # The TPO columns themselves are unchanged because they were
        # computed once before mutation; the only way the trade list
        # could differ is if the executor read mutated H4 close/high/
        # low. We require trades_at_or_before unchanged.
        bucket_t = h4["time"].iloc[i]
        pre = [k for k in base_keys if k[0] <= bucket_t]
        new_pre = [k for k in _trade_keys(new_trades) if k[0] <= bucket_t]
        assert pre == new_pre, (
            f"tpo_value_acceptance: future mutation changed entries "
            f"at or before {bucket_t}")


def test_body_atr_no_lookahead():
    _check_filter_invariant(
        {"type": "body_atr", "min": 0.5, "atr_n": 14},
        "body_atr")


def test_regime_no_lookahead():
    _check_filter_invariant(
        {"type": "regime", "ma_n": 20, "side": "with"},
        "regime")


def test_regime_class_no_lookahead():
    _check_filter_invariant(
        {"type": "regime_class",
         "allow": ["trend", "expansion", "compression", "range", "other"],
         "n_lookback": 8, "atr_window": 50},
        "regime_class")


def test_past_mutation_can_change_decision():
    """Sanity: filters CAN respond to past data, otherwise they'd be inert.
    Mutate an early bar dramatically; expect at least some trades to flip."""
    rng = np.random.default_rng(11)
    h4, m15 = _make_synthetic()
    spec = Spec(
        id="t_past",
        signal={"type": "prev_color"},
        # body_atr depends directly on prior body/ATR, so a past mutation
        # ought to ripple into the trade set.
        filters=[{"type": "body_atr", "min": 0.5, "atr_n": 14}],
        entry={"type": "touch_entry"},
        stop={"type": "none"},
        exit={"type": "h4_close"},
    )
    base_keys = _trade_keys(_run(spec, h4, m15))
    h4_m = h4.copy()
    # Inflate one early bar's range hugely.
    j = 30
    h4_m.loc[j, "high"] = h4_m.loc[j, "high"] + 200.0
    h4_m.loc[j, "low"] = h4_m.loc[j, "low"] - 200.0
    h4_m.loc[j, "close"] = h4_m.loc[j, "open"] + 150.0  # huge body
    new_keys = _trade_keys(_run(spec, h4_m, m15))
    assert base_keys != new_keys, (
        "filter is inert: mutating a prior bar produced identical trades")


if __name__ == "__main__":
    fns = [
        test_atr_percentile_no_lookahead,
        test_pdh_pdl_no_lookahead_inside,
        test_pdh_pdl_no_lookahead_breakout,
        test_atr_distance_from_session_mean_no_lookahead,
        test_tpo_value_acceptance_no_lookahead,
        test_body_atr_no_lookahead,
        test_regime_no_lookahead,
        test_regime_class_no_lookahead,
        test_past_mutation_can_change_decision,
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
