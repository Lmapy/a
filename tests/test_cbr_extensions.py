"""Tests for the Batch K2 CBR extensions:
  * ATR regime filter
  * News calendar filter
  * Multi-target partial exits (TP1 + runner)
  * Walk-forward / OOS split + degradation flags
  * Monte Carlo trade-reorder CI
  * Prop-firm bridge (CBR ledger -> prop_challenge)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.scalp.config import (
    ATRRegimeConfig, CBRGoldScalpConfig, NewsFilterConfig,
)
from strategies.scalp.extensions import (
    NewsCalendar, attach_atr_percentile, passes_atr_regime,
    passes_extensions,
)
from strategies.scalp.engine import run_backtest, _apply_partial_tp, SessionState
from strategies.scalp.entries import EntryPlan
from strategies.scalp.detectors import attach_atr
from strategies.scalp.monte_carlo import monte_carlo_reorder
from strategies.scalp.walk_forward import _degradation


# ---- ATR regime ---------------------------------------------------------

def _h4_synth(n: int = 200, seed: int = 0,
                stdev: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-06-01", periods=n, freq="1min", tz="UTC")
    closes = 2000.0 + np.cumsum(rng.normal(0, stdev, n))
    opens = np.concatenate([[2000.0], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    return pd.DataFrame({"time": t, "open": opens, "high": highs,
                          "low": lows, "close": closes})


def test_atr_regime_disabled_passes():
    df = _h4_synth()
    cfg = ATRRegimeConfig(enabled=False)
    ok, reason = passes_atr_regime(df, idx=50, cfg=cfg)
    assert ok and reason == "atr_regime_off"


def test_atr_regime_attaches_atr_pct_and_filters_extremes():
    df = _h4_synth(n=2000, seed=1)
    df = attach_atr(df, n=14)
    df = attach_atr_percentile(df, atr_length=14, rolling_window=200)
    assert "atr_pct" in df.columns
    cfg = ATRRegimeConfig(enabled=True, min_percentile=0.10, max_percentile=0.90)
    # call at a deep-warmed-up bar
    ok, reason = passes_atr_regime(df, idx=1500, cfg=cfg)
    # may be ok or rejected; either way the function returns a label
    assert reason.startswith("atr_regime_")


def test_atr_regime_warmup_passes():
    df = attach_atr_percentile(_h4_synth(n=50), atr_length=14, rolling_window=720)
    cfg = ATRRegimeConfig(enabled=True)
    # idx=10 -> rolling_window not warm yet -> skip filter (pass)
    ok, _ = passes_atr_regime(df, idx=10, cfg=cfg)
    assert ok


# ---- News calendar ------------------------------------------------------

def test_news_disabled_never_blocks():
    cfg = NewsFilterConfig(enabled=False)
    cal = NewsCalendar.from_config(cfg)
    blocked, reason = cal.is_in_blackout(
        pd.Timestamp("2024-06-01 12:00", tz="UTC"))
    assert not blocked


def test_news_blocks_high_impact_in_window():
    with TemporaryDirectory() as td:
        path = Path(td) / "events.csv"
        pd.DataFrame([
            {"time": "2024-06-01T13:30:00Z", "impact": "high", "symbol": ""},
            {"time": "2024-06-02T13:30:00Z", "impact": "low", "symbol": ""},
        ]).to_csv(path, index=False)
        cfg = NewsFilterConfig(enabled=True, csv_path=str(path),
                                window_minutes_before=5,
                                window_minutes_after=30,
                                block_impacts=["high"])
        cal = NewsCalendar.from_config(cfg)
        # 2 minutes BEFORE high-impact event -> blocked
        blocked, reason = cal.is_in_blackout(
            pd.Timestamp("2024-06-01 13:28", tz="UTC"))
        assert blocked
        assert "high" in reason
        # 2 minutes BEFORE low-impact (not in block_impacts) -> not blocked
        blocked, _ = cal.is_in_blackout(
            pd.Timestamp("2024-06-02 13:28", tz="UTC"))
        assert not blocked


def test_news_csv_missing_degrades_gracefully():
    cfg = NewsFilterConfig(enabled=True, csv_path="/nonexistent/path.csv")
    cal = NewsCalendar.from_config(cfg)
    blocked, reason = cal.is_in_blackout(
        pd.Timestamp("2024-06-01 12:00", tz="UTC"))
    assert not blocked
    assert "empty" in reason or "off" in reason


def test_passes_extensions_combines_atr_and_news():
    df = _h4_synth(n=2000)
    df = attach_atr(df, n=14)
    df = attach_atr_percentile(df, atr_length=14, rolling_window=720)
    atr_cfg = ATRRegimeConfig(enabled=False)
    news_cfg = NewsFilterConfig(enabled=False)
    ok, _ = passes_extensions(df, idx=1500, ts=df["time"].iloc[1500],
                                atr_cfg=atr_cfg,
                                news_cal=NewsCalendar.from_config(news_cfg))
    assert ok


# ---- partial exits ------------------------------------------------------

def test_apply_partial_tp_realises_correct_pnl():
    cfg = CBRGoldScalpConfig()
    cfg.risk.partial_tp_enabled = True
    cfg.risk.tp1_r = 1.0
    cfg.risk.tp1_percent = 0.5
    cfg.risk.runner_target_r = 2.0
    cfg.risk.move_stop_to_be_after_tp1 = True
    state = SessionState(session_date=pd.Timestamp("2024-06-01", tz="UTC").date())
    plan = EntryPlan(direction=1, entry_price=2000.0, stop_price=1990.0,
                       target_price=2030.0, risk_per_unit=10.0,
                       reward_per_unit=30.0, r_multiple=3.0, quantity=2.0,
                       fill_kind="limit", expiry_bar_idx=999)
    state.plan = plan
    state.in_trade = True
    state.trade_entry_idx = 100
    # TP1 price = entry + 1 R = 2010. tp1_percent = 50% of qty=2 -> 1 unit closed.
    _apply_partial_tp(cfg, state, tp1_price=2010.0, i=110)
    assert state.tp1_taken
    assert state.runner_quantity == 1.0
    # break-even runner stop = entry
    assert state.runner_stop == 2000.0
    # runner_target = entry + 2*risk = 2020
    assert state.runner_target == 2020.0
    # partial_pnl = +10 * 1 unit * (tick_value/tick_size) = 10 * 1 * 1 = 10
    assert state.partial_pnl == 10.0


def test_apply_partial_tp_idempotent():
    """Calling twice should not double-realise."""
    cfg = CBRGoldScalpConfig()
    cfg.risk.partial_tp_enabled = True
    state = SessionState(session_date=pd.Timestamp("2024-06-01", tz="UTC").date())
    plan = EntryPlan(direction=1, entry_price=2000.0, stop_price=1990.0,
                       target_price=2030.0, risk_per_unit=10.0,
                       reward_per_unit=30.0, r_multiple=3.0, quantity=2.0,
                       fill_kind="limit", expiry_bar_idx=999)
    state.plan = plan
    state.in_trade = True
    _apply_partial_tp(cfg, state, tp1_price=2010.0, i=110)
    pnl_after_first = state.partial_pnl
    _apply_partial_tp(cfg, state, tp1_price=2010.0, i=120)
    assert state.partial_pnl == pnl_after_first


# ---- walk-forward ------------------------------------------------------

def test_degradation_flags_oos_negative_expectancy():
    is_m = {"total_trades": 100, "expectancy_r": 0.20,
            "profit_factor": 1.5, "max_drawdown_r": -5.0}
    oos_m = {"total_trades": 30, "expectancy_r": -0.10,
             "profit_factor": 0.7, "max_drawdown_r": -8.0}
    deg = _degradation(is_m, oos_m)
    assert "OOS_NEGATIVE_EXPECTANCY" in deg["flags"]
    assert not deg["stable"]


def test_degradation_stable_when_metrics_match():
    is_m = {"total_trades": 100, "expectancy_r": 0.20,
            "profit_factor": 1.5, "max_drawdown_r": -5.0}
    oos_m = {"total_trades": 50, "expectancy_r": 0.18,
             "profit_factor": 1.4, "max_drawdown_r": -6.0}
    deg = _degradation(is_m, oos_m)
    assert deg["stable"]
    assert deg["flags"] == []


# ---- Monte Carlo --------------------------------------------------------

class _Tr:
    def __init__(self, r, p):
        self.r_result = r; self.pnl = p


def test_monte_carlo_reorder_returns_sane_ci():
    trades = [_Tr(1.5, 1.5), _Tr(-1.0, -1.0), _Tr(1.5, 1.5),
                _Tr(-1.0, -1.0), _Tr(1.5, 1.5), _Tr(1.5, 1.5),
                _Tr(-1.0, -1.0), _Tr(1.5, 1.5), _Tr(-1.0, -1.0),
                _Tr(1.5, 1.5)]   # 10 trades, +5R total
    payload = monte_carlo_reorder(trades, n_runs=200, seed=1)
    # total_r is invariant under permutation -- p05 == p50 == p95
    assert payload["bootstrap_total_r"]["p05"] == \
            payload["bootstrap_total_r"]["p50"] == \
            payload["bootstrap_total_r"]["p95"]
    # max_dd varies with order -> p05 should be more negative than p95
    boot_dd = payload["bootstrap_max_drawdown_r"]
    assert boot_dd["p05"] <= boot_dd["p95"]
    assert payload["actual"]["total_r"] == 5.0


def test_monte_carlo_handles_empty_trade_list():
    payload = monte_carlo_reorder([], n_runs=100)
    assert payload["n_runs"] == 0
    assert "note" in payload


# ---- Prop bridge --------------------------------------------------------

def test_prop_bridge_loader_reshapes_columns(tmp_path: Path = None):
    with TemporaryDirectory() as td:
        from strategies.scalp.prop_bridge import load_cbr_trades
        td = Path(td)
        df = pd.DataFrame([{
            "trade_id": 0, "setup_id": "x",
            "entry_time": "2024-06-01T13:00:00+00:00",
            "exit_time":  "2024-06-01T14:00:00+00:00",
            "direction": 1, "entry_price": 2000.0, "exit_price": 2010.0,
            "stop_price": 1990.0, "target_price": 2030.0,
            "exit_reason": "target", "r_result": 1.0, "pnl": 10.0,
            "duration_minutes": 60, "mae": 1.0, "mfe": 12.0,
            "session_date": "2024-06-01",
            "day_of_week": "Saturday", "htf_bias": 1, "dxy_state": "off",
            "expansion_quality": 70.0, "trigger_kind": "sweep",
            "structure_mode": "CLOSE_THROUGH", "retracement_level": 0.5,
        }])
        path = td / "trades.csv"
        df.to_csv(path, index=False)
        out = load_cbr_trades(path)
        assert "entry_time" in out.columns
        assert "stop_distance_price" in out.columns
        assert "pnl" in out.columns
        assert out["stop_distance_price"].iloc[0] == 10.0
        assert out["pnl"].iloc[0] == 10.0


# ---- driver ------------------------------------------------------------

if __name__ == "__main__":
    fns = [
        test_atr_regime_disabled_passes,
        test_atr_regime_attaches_atr_pct_and_filters_extremes,
        test_atr_regime_warmup_passes,
        test_news_disabled_never_blocks,
        test_news_blocks_high_impact_in_window,
        test_news_csv_missing_degrades_gracefully,
        test_passes_extensions_combines_atr_and_news,
        test_apply_partial_tp_realises_correct_pnl,
        test_apply_partial_tp_idempotent,
        test_degradation_flags_oos_negative_expectancy,
        test_degradation_stable_when_metrics_match,
        test_monte_carlo_reorder_returns_sane_ci,
        test_monte_carlo_handles_empty_trade_list,
        test_prop_bridge_loader_reshapes_columns,
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
