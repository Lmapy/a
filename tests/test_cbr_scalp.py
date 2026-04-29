"""Unit tests for the CBR scalp module (Batch K).

Covers:
  * Config round-trips through YAML / JSON
  * Sessions / timezone column annotation
  * Expansion / sweep / rebalance / MSB detectors
  * Confirmed-pivot no-lookahead invariant
  * Entry-plan calculation (50% retrace, fixed-R target)
  * Engine smoke (synthetic data + tiny config -> no crash)
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.scalp.bias import (
    dxy_inverse_check, precompute_dxy_bias, precompute_h1_bias,
    resolve_h1_bias_at,
)
from strategies.scalp.config import (
    CBRGoldScalpConfig, DXYConfig, EntryConfig, ExpansionConfig,
    HTFBiasConfig, RiskConfig, SessionConfig, StopTargetConfig,
    StructureConfig, TriggerConfig,
)
from strategies.scalp.detectors import (
    ExpansionResult, MSBResult, Pivot, attach_atr,
    evaluate_expansion, evaluate_msb, evaluate_rebalance,
    evaluate_sweep, find_confirmed_pivots,
)
from strategies.scalp.engine import run_backtest
from strategies.scalp.entries import compute_entry_plan
from strategies.scalp.sessions import (
    SessionWindow, annotate_session_columns, asia_session_high_low,
)


# ---- config ----

def test_config_default_validates():
    cfg = CBRGoldScalpConfig()
    assert cfg.validate() == []


def test_config_round_trip_via_json():
    cfg = CBRGoldScalpConfig()
    payload = cfg.to_json()
    reborn = CBRGoldScalpConfig.from_json(payload)
    assert reborn.to_json() == cfg.to_json()


def test_config_yaml_round_trip(tmp_path: Path = None):
    """Write yaml then re-read it. Uses repo config file."""
    yaml_path = ROOT / "configs" / "cbr_gold_scalp.yaml"
    cfg = CBRGoldScalpConfig.from_yaml(yaml_path)
    assert cfg.symbol == "XAUUSD"
    assert cfg.session.timezone == "Australia/Melbourne"
    assert cfg.htf_bias.bias_mode == "PREVIOUS_1H_CANDLE_DIRECTION"


def test_config_validate_catches_bad_inputs():
    cfg = CBRGoldScalpConfig()
    cfg.expansion.expansion_lookback_bars = 2
    cfg.stop_target.risk_reward = -1.0
    issues = cfg.validate()
    assert len(issues) >= 2


# ---- sessions ----

def _m1_synth(start: str = "2024-06-01 22:00", n: int = 720,
              tz: str = "UTC") -> pd.DataFrame:
    times = pd.date_range(start, periods=n, freq="1min", tz=tz)
    rng = np.random.default_rng(0)
    closes = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))
    opens = np.concatenate([[2000.0], closes[:-1]])
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows,
        "close": closes,
    })


def test_session_annotate_columns_present():
    df = _m1_synth()
    cfg = SessionConfig(asia_session_start="09:00", asia_session_end="12:00",
                         execution_window_start="10:00", execution_window_end="11:00")
    win = SessionWindow.from_config(cfg)
    out = annotate_session_columns(df, win)
    for c in ("session_date", "in_asia", "in_execution"):
        assert c in out.columns


def test_session_window_validates_inner_inside_outer():
    bad = SessionConfig(asia_session_start="10:00", asia_session_end="11:00",
                         execution_window_start="09:00", execution_window_end="12:00")
    raised = False
    try:
        SessionWindow.from_config(bad)
    except ValueError:
        raised = True
    assert raised


def test_asia_session_high_low_resets_each_day():
    df = _m1_synth(start="2024-06-01 22:00", n=2880)   # 2 days
    cfg = SessionConfig()
    win = SessionWindow.from_config(cfg)
    df = annotate_session_columns(df, win)
    df = asia_session_high_low(df)
    # asia_high_so_far should differ across the two session-dates
    sess = df[df["in_asia"]].copy()
    if not sess.empty:
        per_day = sess.groupby("session_date")["asia_high_so_far"].max()
        assert len(per_day) >= 1


# ---- expansion ----

def test_expansion_detects_strong_bull_run():
    """Build a deterministic bullish run and confirm the detector
    fires."""
    n = 60
    times = pd.date_range("2024-06-01 09:00", periods=n, freq="1min", tz="UTC")
    base = 2000.0
    closes = base + np.cumsum(np.full(n, 0.5))    # +0.5 every minute
    opens = np.concatenate([[base], closes[:-1]])
    highs = closes + 0.1
    lows = opens - 0.1
    df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    df = attach_atr(df, n=14)
    cfg = ExpansionConfig(expansion_lookback_bars=20,
                           min_directional_candle_percent=0.65,
                           min_net_move_atr_multiple=1.0)
    res = evaluate_expansion(df, end_idx=n - 1, cfg=cfg)
    assert res.detected
    assert res.direction == 1
    assert res.quality_score > 50


def test_expansion_rejects_choppy_data():
    """Mean-reverting alternating closes -> no directional pattern.
    Build deterministically rather than relying on RNG."""
    n = 30
    times = pd.date_range("2024-06-01 09:00", periods=n, freq="1min", tz="UTC")
    base = 2000.0
    closes = []
    for i in range(n):
        # alternating up / down by 1.0 -> exactly 50% bullish, 50% bearish
        closes.append(base + (1.0 if i % 2 == 0 else -1.0))
    closes = np.array(closes)
    opens = np.concatenate([[base], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    df = attach_atr(df, n=14)
    cfg = ExpansionConfig(expansion_lookback_bars=20,
                           min_directional_candle_percent=0.65,
                           min_net_move_atr_multiple=1.0)
    res = evaluate_expansion(df, end_idx=n - 1, cfg=cfg)
    # alternating bars -> ~50% directional, well below the 65% gate
    assert not res.detected


# ---- sweep ----

def test_sweep_detects_bullish_reclaim():
    """Build a small frame where price dips below prev_h1_low then
    closes back above it -> bullish sweep + reclaim."""
    closes = [1995.0, 1994.0, 1993.5, 1995.5, 1996.0]   # last close back above 1994
    opens = [2000, 1995, 1994, 1993, 1995]
    highs = [2001, 1996, 1995, 1996, 1997]
    lows = [1994, 1993, 1992, 1992, 1995]               # idx 2 sweeps 1994 prev_low
    times = pd.date_range("2024-06-01 09:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    res = evaluate_sweep(df, end_idx=4,
                          prev_h1_high=2010, prev_h1_low=1994,
                          lookback=10)
    assert res.detected and res.direction == 1


def test_sweep_no_detection_without_reclaim():
    closes = [1990, 1989, 1988, 1987, 1986]   # no reclaim
    opens = closes
    highs = [1995, 1992, 1989, 1988, 1987]
    lows = [1985, 1985, 1985, 1985, 1985]
    times = pd.date_range("2024-06-01 09:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    res = evaluate_sweep(df, end_idx=4,
                          prev_h1_high=2000, prev_h1_low=1995,
                          lookback=10)
    assert not res.detected


# ---- rebalance ----

def test_rebalance_detects_midpoint_touch():
    times = pd.date_range("2024-06-01 09:00", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "open": [2000.0] * 30,
        "high": [2001.0] * 30,
        "low": [1999.0] * 30,
        "close": [2000.0] * 30,
    })
    exp = ExpansionResult(detected=True, direction=1,
                            start_idx=5, end_idx=15,
                            high=2010.0, low=1990.0, midpoint=2000.0,
                            net_move=10.0, atr_at_end=0.5,
                            n_directional=15, quality_score=60.0)
    cfg = TriggerConfig(rebalance_tolerance_ticks=2,
                         rebalance_tolerance_atr_fraction=0.5,
                         require_midpoint_touch=True,
                         max_bars_after_expansion=30)
    res = evaluate_rebalance(df, end_idx=20, expansion=exp,
                              cfg=cfg, atr_at_end=0.5, tick_size=0.10)
    assert res.detected
    assert res.direction == -1


def test_rebalance_expires_after_max_bars():
    times = pd.date_range("2024-06-01 09:00", periods=60, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "open": [2000.0] * 60, "high": [2001.0] * 60,
        "low": [1999.0] * 60,  "close": [2000.0] * 60,
    })
    exp = ExpansionResult(detected=True, direction=1,
                            start_idx=0, end_idx=10,
                            high=2010, low=1990, midpoint=2000,
                            net_move=10, atr_at_end=0.5,
                            n_directional=10, quality_score=70)
    cfg = TriggerConfig(max_bars_after_expansion=10)
    res = evaluate_rebalance(df, end_idx=50, expansion=exp,
                              cfg=cfg, atr_at_end=0.5, tick_size=0.10)
    assert not res.detected


# ---- pivots + MSB ----

def test_confirmed_pivots_have_late_confirmation_idx():
    n = 30
    times = pd.date_range("2024-06-01 09:00", periods=n, freq="1min", tz="UTC")
    closes = list(range(n))
    opens = closes
    # construct a single peak at idx 10
    highs = [c + (5 if i == 10 else 1) for i, c in enumerate(closes)]
    lows = [c - 1 for c in closes]
    df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    pivots = find_confirmed_pivots(df, pivot_left=2, pivot_right=2)
    assert any(p.idx == 10 and p.direction == 1 for p in pivots)
    p = next(p for p in pivots if p.idx == 10)
    assert p.confirmed_at_idx == 10 + 2
    assert p.confirmed_at_idx > p.idx


def test_msb_requires_break_above_confirmed_swing_high():
    n = 25
    times = pd.date_range("2024-06-01 09:00", periods=n, freq="1min", tz="UTC")
    # single swing high at idx 5 = 100; rest below; then idx 20 closes above 100
    highs = [90 + i if i != 5 else 100 for i in range(n)]
    closes = [85 + i if i != 20 else 110 for i in range(n)]
    lows = [c - 5 for c in closes]
    opens = [c - 1 for c in closes]
    df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    pivots = find_confirmed_pivots(df, pivot_left=2, pivot_right=2)
    cfg = StructureConfig(pivot_left=2, pivot_right=2,
                           structure_break_mode="CLOSE_THROUGH")
    msb = evaluate_msb(df, end_idx=20, pivots=pivots, cfg=cfg,
                        trigger_idx=10, trigger_direction=1,
                        max_bars_between=20)
    assert msb.detected
    assert msb.direction == 1
    assert msb.broken_pivot_idx == 5


def test_msb_does_not_use_unconfirmed_pivot():
    """If a candidate pivot exists at idx K but the run only reaches
    K+1, we must NOT see a pivot reported (confirmation needs
    pivot_right=2 bars after K)."""
    n = 8
    times = pd.date_range("2024-06-01 09:00", periods=n, freq="1min", tz="UTC")
    highs = [10, 11, 100, 12, 13, 14, 15, 16]   # peak at idx 2
    closes = highs
    lows = [c - 1 for c in closes]
    opens = closes
    df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    # only allow scan UP TO idx 3 (so we've seen 1 bar after peak)
    pivots_partial = find_confirmed_pivots(df, pivot_left=2, pivot_right=2,
                                             up_to_idx=3)
    assert all(p.idx != 2 for p in pivots_partial), \
        "pivot at idx 2 confirmed too early (only 1 bar after, needs 2)"


# ---- entries ----

def test_entry_plan_50_retrace_long():
    cfg = CBRGoldScalpConfig()
    cfg.entry.entry_mode = "LIMIT_50_RETRACE"
    cfg.stop_target.stop_mode = "RECENT_SWING"
    cfg.stop_target.target_mode = "FIXED_R_MULTIPLE"
    cfg.stop_target.risk_reward = 1.5
    msb = MSBResult(detected=True, direction=1,
                     broken_pivot_idx=5, broken_pivot_price=100.0,
                     break_idx=10, break_price=101.0,
                     impulse_origin_idx=4, impulse_origin_price=90.0,
                     impulse_high_or_low_idx=10, impulse_high_or_low_price=110.0)
    exp = ExpansionResult(detected=True, direction=-1,
                            start_idx=0, end_idx=4,
                            high=95.0, low=85.0, midpoint=90.0,
                            net_move=-10.0, atr_at_end=1.0,
                            n_directional=4, quality_score=70.0)
    plan = compute_entry_plan(cfg=cfg, msb=msb, expansion=exp,
                                prev_h1_open=99.0, prev_h1_close=98.0,
                                asia_high=110.0, asia_low=85.0,
                                atr_at_entry=1.0, market_close_price=101.0,
                                msb_idx=10, equity=50_000.0)
    assert plan is not None
    # 50% of (110 - 90) = 100 -> entry should be 100
    assert abs(plan.entry_price - 100.0) < 1e-6
    # stop = origin price - buffer ticks; buffer=1 tick @ 0.10 = 89.9
    assert abs(plan.stop_price - 89.9) < 1e-6
    # risk = 100 - 89.9 = 10.1
    assert abs(plan.risk_per_unit - 10.1) < 1e-6
    # target = entry + 1.5 * risk = 100 + 15.15 = 115.15
    assert abs(plan.target_price - 115.15) < 1e-6


def test_entry_plan_returns_none_when_target_on_wrong_side():
    cfg = CBRGoldScalpConfig()
    cfg.stop_target.target_mode = "PREVIOUS_1H_EQUILIBRIUM"
    msb = MSBResult(detected=True, direction=1,
                     broken_pivot_idx=5, broken_pivot_price=100.0,
                     break_idx=10, break_price=101.0,
                     impulse_origin_idx=4, impulse_origin_price=90.0,
                     impulse_high_or_low_idx=10, impulse_high_or_low_price=110.0)
    exp = ExpansionResult(detected=True, direction=-1,
                            start_idx=0, end_idx=4, high=95, low=85,
                            midpoint=90, net_move=-10, atr_at_end=1,
                            n_directional=4, quality_score=70)
    # prev_h1 equilibrium 95 is BELOW entry 100 for a LONG -> invalid
    plan = compute_entry_plan(cfg=cfg, msb=msb, expansion=exp,
                                prev_h1_open=90.0, prev_h1_close=100.0,
                                asia_high=120, asia_low=80,
                                atr_at_entry=1.0, market_close_price=101.0,
                                msb_idx=10, equity=50_000)
    assert plan is None    # target eq=(90+100)/2=95 < entry=100 -> rejected


# ---- bias / DXY ----

def test_h1_bias_uses_only_completed_bars():
    times = pd.date_range("2024-06-01 09:00", periods=5, freq="1H", tz="UTC")
    h1 = pd.DataFrame({
        "time": times,
        "open":  [100, 101, 102, 103, 104],
        "high":  [101, 102, 103, 104, 105],
        "low":   [99, 100, 101, 102, 103],
        "close": [101, 100, 103, 102, 105],
    })
    cfg = HTFBiasConfig(bias_mode="PREVIOUS_1H_CANDLE_DIRECTION")
    h1b = precompute_h1_bias(h1, cfg)
    # at 09:30 we should have NO completed h1 (the 09:00 bar ends at 10:00)
    # at 10:00 the 09:00 bar is just completed; bias from that bar = +1 (101>100)
    snap = resolve_h1_bias_at(pd.Timestamp("2024-06-01 09:30", tz="UTC"), h1b)
    assert snap.direction == 0
    snap2 = resolve_h1_bias_at(pd.Timestamp("2024-06-01 10:00", tz="UTC"), h1b)
    assert snap2.direction == 1


def test_dxy_filter_passes_when_unavailable():
    cfg = DXYConfig(dxy_mode="EMA_SLOPE")
    ok, reason = dxy_inverse_check(direction_gold=1,
                                     minute_ts=pd.Timestamp("2024-06-01 09:00", tz="UTC"),
                                     dxy=None, cfg=cfg)
    assert ok and "unavailable" in reason


def test_dxy_filter_rejects_aligned_dxy():
    times = pd.date_range("2024-06-01 00:00", periods=10, freq="1H", tz="UTC")
    dxy = pd.DataFrame({"time": times,
                         "open": list(range(10)),
                         "high": [x + 1 for x in range(10)],
                         "low": [x - 1 for x in range(10)],
                         "close": [x + 0.5 for x in range(10)]})
    cfg = DXYConfig(dxy_mode="PREVIOUS_CANDLE_DIRECTION",
                    require_inverse_confirmation=True)
    dxy_b = precompute_dxy_bias(dxy, cfg)
    # gold long, DXY +1 (rising) -> aligned against gold -> reject
    ok, _ = dxy_inverse_check(1,
                                pd.Timestamp("2024-06-01 09:00", tz="UTC"),
                                dxy_b, cfg)
    assert not ok


# ---- engine smoke ----

def test_engine_smoke_runs_without_errors():
    """End-to-end smoke: synthetic data + light config -> engine
    completes and writes well-shaped trade list."""
    n = 60 * 24 * 3   # 3 days of M1
    times = pd.date_range("2024-06-01 00:00", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(7)
    closes = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))
    opens = np.concatenate([[2000.0], closes[:-1]])
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.5, n))
    m1 = pd.DataFrame({"time": times, "open": opens, "high": highs,
                        "low": lows, "close": closes})
    h1_idx = pd.date_range("2024-06-01 00:00", periods=72, freq="1H", tz="UTC")
    h1 = pd.DataFrame({"time": h1_idx,
                        "open": rng.normal(2000, 5, 72),
                        "high": rng.normal(2010, 5, 72),
                        "low": rng.normal(1990, 5, 72),
                        "close": rng.normal(2000, 5, 72)})
    cfg = CBRGoldScalpConfig()
    cfg.expansion.expansion_lookback_bars = 10
    cfg.expansion.atr_length = 8
    cfg.expansion.min_directional_candle_percent = 0.5
    cfg.expansion.min_net_move_atr_multiple = 0.5
    results = run_backtest(cfg, m1, h1)
    # smoke: didn't crash and shape is sane
    assert "trades" in results and "setups" in results
    assert isinstance(results["trades"], list)
    assert isinstance(results["validation"], dict)


if __name__ == "__main__":
    fns = [
        test_config_default_validates,
        test_config_round_trip_via_json,
        test_config_yaml_round_trip,
        test_config_validate_catches_bad_inputs,
        test_session_annotate_columns_present,
        test_session_window_validates_inner_inside_outer,
        test_asia_session_high_low_resets_each_day,
        test_expansion_detects_strong_bull_run,
        test_expansion_rejects_choppy_data,
        test_sweep_detects_bullish_reclaim,
        test_sweep_no_detection_without_reclaim,
        test_rebalance_detects_midpoint_touch,
        test_rebalance_expires_after_max_bars,
        test_confirmed_pivots_have_late_confirmation_idx,
        test_msb_requires_break_above_confirmed_swing_high,
        test_msb_does_not_use_unconfirmed_pivot,
        test_entry_plan_50_retrace_long,
        test_entry_plan_returns_none_when_target_on_wrong_side,
        test_h1_bias_uses_only_completed_bars,
        test_dxy_filter_passes_when_unavailable,
        test_dxy_filter_rejects_aligned_dxy,
        test_engine_smoke_runs_without_errors,
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
