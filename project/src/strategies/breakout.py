from __future__ import annotations

import pandas as pd

from src.strategies.common import base_signal_frame


def generate_breakout_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = base_signal_frame(df)
    squeeze_q = df["bb_width_20_2"].rolling(cfg["squeeze_lookback"]).quantile(cfg["squeeze_pctile"])
    squeeze = df["bb_width_20_2"] < squeeze_q
    long_cond = squeeze & (df["close"] > df["donchian_high_20"].shift(1)) & (df["volume"] > cfg["volume_multiple"] * df["rolling_volume_mean"])
    short_cond = squeeze & (df["close"] < df["donchian_low_20"].shift(1)) & (df["volume"] > cfg["volume_multiple"] * df["rolling_volume_mean"])
    out.loc[long_cond, "long_entry"] = True
    out.loc[short_cond, "short_entry"] = True
    out.loc[long_cond, "stop_price"] = df["close"] - cfg["initial_stop_atr"] * df["atr_14"]
    out.loc[short_cond, "stop_price"] = df["close"] + cfg["initial_stop_atr"] * df["atr_14"]
    out.loc[long_cond | short_cond, "tag"] = "volatility_expansion_breakout"
    return out
