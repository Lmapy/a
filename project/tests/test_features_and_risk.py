from __future__ import annotations

import pandas as pd

from src.backtest.engine import _risk_position_size
from src.features.engineer import build_features


def test_risk_position_size_respects_leverage_cap() -> None:
    qty = _risk_position_size(equity=1000, risk_pct=0.01, entry=100, stop=99, leverage_cap=2)
    assert qty * 100 <= 2000 + 1e-9


def test_feature_builder_emits_required_columns() -> None:
    ts = pd.date_range("2024-01-01", periods=300, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "funding_rate": 0.0,
            "open_interest": 1.0,
            "mark_price": 100.4,
            "index_price": 100.3,
            "exchange": "bybit",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
        }
    )
    out = build_features(df)
    for col in ["atr_14", "bb_width_20_2", "donchian_high_20", "vwap_zscore", "rsi_14", "prev_day_high", "asia_high", "sweep_low_flag"]:
        assert col in out.columns
