from __future__ import annotations

import pandas as pd


def base_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["timestamp", "close", "atr_14", "session_vwap", "vwap_zscore", "rsi_14", "rolling_return_vol"]].copy()
    out["long_entry"] = False
    out["short_entry"] = False
    out["stop_price"] = pd.NA
    out["target_price"] = pd.NA
    out["tag"] = ""
    return out
