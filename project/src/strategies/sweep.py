from __future__ import annotations

import pandas as pd

from src.strategies.common import base_signal_frame


def generate_sweep_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = base_signal_frame(df)
    swept_low_level = df[["prev_day_low", "asia_low"]].min(axis=1)
    swept_high_level = df[["prev_day_high", "asia_high"]].max(axis=1)

    long_cond = (
        (df["low"] < swept_low_level)
        & (df["close"] > swept_low_level)
        & (df["lower_wick_body_ratio"] > cfg["wick_body_ratio_min"])
    )
    short_cond = (
        (df["high"] > swept_high_level)
        & (df["close"] < swept_high_level)
        & (df["upper_wick_body_ratio"] > cfg["wick_body_ratio_min"])
    )

    if cfg.get("use_momentum_exhaustion_filter", False):
        long_cond &= df["rsi_14"] <= cfg.get("momentum_rsi_thresh", 30)
        short_cond &= df["rsi_14"] >= (100 - cfg.get("momentum_rsi_thresh", 30))

    out.loc[long_cond, "long_entry"] = True
    out.loc[short_cond, "short_entry"] = True
    out.loc[long_cond, "stop_price"] = df["low"] - cfg["stop_buffer_atr"] * df["atr_14"]
    out.loc[short_cond, "stop_price"] = df["high"] + cfg["stop_buffer_atr"] * df["atr_14"]
    out.loc[long_cond | short_cond, "target_price"] = df["session_vwap"]
    out.loc[long_cond, "tag"] = "sweep_reversal_long"
    out.loc[short_cond, "tag"] = "sweep_reversal_short"
    out["sweep_level"] = pd.NA
    out.loc[long_cond, "sweep_level"] = swept_low_level[long_cond]
    out.loc[short_cond, "sweep_level"] = swept_high_level[short_cond]
    return out
