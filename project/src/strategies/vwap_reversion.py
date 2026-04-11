from __future__ import annotations

import pandas as pd

from src.strategies.common import base_signal_frame


def generate_vwap_reversion_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = base_signal_frame(df)
    long_cond = (df["vwap_zscore"] < cfg["z_long"]) & (df["rsi_14"] < cfg["rsi_long"])
    short_cond = (df["vwap_zscore"] > cfg["z_short"]) & (df["rsi_14"] > cfg["rsi_short"])

    if cfg.get("disable_in_high_vol_trend", True):
        thresh = df["rolling_return_vol"].rolling(500).quantile(cfg.get("high_vol_quantile", 0.8))
        regime_ok = df["rolling_return_vol"] <= thresh
        long_cond &= regime_ok
        short_cond &= regime_ok

    out.loc[long_cond, "long_entry"] = True
    out.loc[short_cond, "short_entry"] = True
    out.loc[long_cond | short_cond, "target_price"] = df["session_vwap"]
    out.loc[long_cond, "stop_price"] = df["close"] - cfg["stop_sigma"] * df["close"].rolling(120).std()
    out.loc[short_cond, "stop_price"] = df["close"] + cfg["stop_sigma"] * df["close"].rolling(120).std()
    out.loc[long_cond | short_cond, "tag"] = "vwap_zscore_mean_reversion"
    return out
