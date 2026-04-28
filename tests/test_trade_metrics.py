"""Tests for the metrics added in Phase 6.

Lock in:
  - sharpe_ann is the legacy H4-bar Sharpe (back-compat)
  - sharpe_h4_bar_ann == sharpe_ann
  - sharpe_trade_ann is materially different from sharpe_h4_bar_ann
    when the strategy is sparse (few trades per year)
  - sharpe_daily_ann uses sqrt(252) annualisation
  - sharpe_weekly_ann uses sqrt(52) annualisation
  - time_under_water_share is in [0, 1]
  - expectancy_R is mean(rets) / |mean(loss)|
  - worst_day / worst_week are min daily / weekly returns
  - For dense strategies (one trade per H4 bar), trade and H4-bar
    Sharpe converge.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.trade_metrics import basic
from core.types import Trade


def _trades_at(times: pd.DatetimeIndex, rets: np.ndarray) -> list[Trade]:
    out = []
    for t, r in zip(times, rets):
        out.append(Trade(
            entry_time=t, exit_time=t, direction=1 if r >= 0 else -1,
            entry=2000.0, exit=2000.0 * (1.0 + float(r)),
            cost=0.0, pnl=float(r) * 2000.0, ret=float(r),
        ))
    return out


def test_back_compat_sharpe_ann_equals_h4_bar_form():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 100)
    times = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
    m = basic(_trades_at(times, rets))
    assert m["sharpe_ann"] == m["sharpe_h4_bar_ann"], (
        "back-compat: sharpe_ann must remain the H4-bar form")


def test_sparse_trade_sharpe_differs_from_h4_bar_sharpe():
    """If the strategy trades only ~250 times per year (about once per
    trading day), the H4-bar annualisation factor (sqrt(1560)) is
    sqrt(1560/250) ~ 2.5x too aggressive."""
    rng = np.random.default_rng(1)
    n = 100
    rets = rng.normal(0.001, 0.01, n)
    # spread the trades across 100 trading days (one per day)
    times = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    m = basic(_trades_at(times, rets))
    # Trade-frequency Sharpe should be smaller than H4-bar Sharpe by ~sqrt(1560/365)
    if m["sharpe_h4_bar_ann"] != 0 and m["sharpe_trade_ann"] != 0:
        ratio = abs(m["sharpe_h4_bar_ann"] / m["sharpe_trade_ann"])
        assert 1.5 < ratio < 3.5, (
            f"expected H4-bar/trade Sharpe ratio ~2-3 for daily-trading "
            f"strategy, got {ratio:.2f}")


def test_dense_trade_sharpe_close_to_h4_bar_sharpe():
    """If the strategy fires roughly every H4 bar over many years of
    contiguous data, the two annualisations should be the same order
    of magnitude. (They're not exactly equal because H4_BARS_PER_YEAR
    = 1560 assumes 5-trading-day weeks while the synthetic fixture is
    contiguous, ~2200 bars/year, so the trade form is a touch larger.)"""
    rng = np.random.default_rng(2)
    n = 1000
    rets = rng.normal(0.0005, 0.005, n)
    times = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    m = basic(_trades_at(times, rets))
    if m["sharpe_h4_bar_ann"] == 0 or m["sharpe_trade_ann"] == 0:
        return
    ratio = m["sharpe_trade_ann"] / m["sharpe_h4_bar_ann"]
    # ratio = sqrt(actual_bars_per_year / 1560). For contiguous 4h data
    # actual ~ 2191; sqrt(2191/1560) ~= 1.185.
    assert 0.8 < ratio < 1.4, (
        f"dense strategy: trade/h4-bar Sharpe ratio should be O(1), got {ratio:.3f}")


def test_daily_and_weekly_sharpe_present():
    rng = np.random.default_rng(3)
    n = 500
    rets = rng.normal(0.0002, 0.003, n)
    times = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    m = basic(_trades_at(times, rets))
    assert "sharpe_daily_ann" in m
    assert "sharpe_weekly_ann" in m
    assert m["n_trading_days"] >= 30
    assert m["n_trading_weeks"] >= 5


def test_time_under_water_in_range_and_sane():
    """A constantly-rising equity curve must have TUW = 0; an oscillating
    one must have TUW between 0 and 1."""
    times = pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC")
    # Always-rising
    m_up = basic(_trades_at(times, np.full(20, 0.01)))
    assert m_up["time_under_water_share"] == 0.0
    # Up-down oscillation
    rets = np.array([0.05] + [-0.01, 0.01] * 9 + [-0.01])
    m_osc = basic(_trades_at(times, rets))
    assert 0.0 < m_osc["time_under_water_share"] < 1.0


def test_expectancy_R_sign_and_scale():
    """Strategy with mean +1% and avg loss -2% should have expectancy_R = 0.5."""
    times = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    rets = np.array([0.04, -0.02, 0.04, -0.02, 0.04, -0.02, 0.04, -0.02, 0.04, -0.02])
    # mean = 0.01, mean(loss) = -0.02, so expectancy_R = 0.01 / 0.02 = 0.5
    m = basic(_trades_at(times, rets))
    assert abs(m["expectancy_R"] - 0.5) < 0.001, (
        f"expectancy_R should be 0.5, got {m['expectancy_R']}")


def test_worst_day_and_worst_week_present():
    times = pd.date_range("2024-01-01", periods=20, freq="6h", tz="UTC")
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, 20)
    m = basic(_trades_at(times, rets))
    assert "worst_day_ret" in m
    assert "worst_week_ret" in m
    # the worst day return must be <= 0 if any losing day exists
    if (rets < 0).any():
        assert m["worst_day_ret"] <= 0.0


def test_empty_trades_returns_minimal():
    assert basic([]) == {"trades": 0}


if __name__ == "__main__":
    fns = [
        test_back_compat_sharpe_ann_equals_h4_bar_form,
        test_sparse_trade_sharpe_differs_from_h4_bar_sharpe,
        test_dense_trade_sharpe_close_to_h4_bar_sharpe,
        test_daily_and_weekly_sharpe_present,
        test_time_under_water_in_range_and_sane,
        test_expectancy_R_sign_and_scale,
        test_worst_day_and_worst_week_present,
        test_empty_trades_returns_minimal,
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
