from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import save_parquet

LOGGER = logging.getLogger(__name__)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _session_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["timestamp"].dt.floor("D")
    prev = out.groupby("date").agg(prev_day_high=("high", "max"), prev_day_low=("low", "min")).shift(1)
    out = out.merge(prev, on="date", how="left")

    hour = out["timestamp"].dt.hour
    asia = out[hour.between(0, 7)]
    lon = out[hour.between(8, 15)]

    asia_lv = asia.groupby("date").agg(asia_high=("high", "max"), asia_low=("low", "min"))
    lon_lv = lon.groupby("date").agg(london_high=("high", "max"), london_low=("low", "min"))
    out = out.merge(asia_lv, on="date", how="left").merge(lon_lv, on="date", how="left")
    return out.drop(columns=["date"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp")
    out["ret"] = out["close"].pct_change()
    out["atr_14"] = _atr(out, 14)
    ma = out["close"].rolling(20).mean()
    sd = out["close"].rolling(20).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    out["bb_width_20_2"] = (upper - lower) / ma
    out["donchian_high_20"] = out["high"].rolling(20).max()
    out["donchian_low_20"] = out["low"].rolling(20).min()
    out["rolling_volume_mean"] = out["volume"].rolling(20).mean()
    typical = (out["high"] + out["low"] + out["close"]) / 3
    session_key = out["timestamp"].dt.floor("D")
    out["session_vwap"] = (typical * out["volume"]).groupby(session_key).cumsum() / out["volume"].groupby(session_key).cumsum().replace(0, np.nan)
    out["rolling_vwap"] = (typical * out["volume"]).rolling(120).sum() / out["volume"].rolling(120).sum().replace(0, np.nan)
    out["vwap_zscore"] = (out["close"] - out["rolling_vwap"]) / out["close"].rolling(120).std()
    out["rsi_14"] = _rsi(out["close"], 14)
    out["rolling_return_vol"] = out["ret"].rolling(60).std()
    out = _session_levels(out)
    out["sweep_low_flag"] = (out["low"] < out["prev_day_low"]) | (out["low"] < out["asia_low"])
    out["sweep_high_flag"] = (out["high"] > out["prev_day_high"]) | (out["high"] > out["asia_high"])
    body = (out["close"] - out["open"]).abs().replace(0, np.nan)
    out["lower_wick_body_ratio"] = (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0) / body
    out["upper_wick_body_ratio"] = (out["high"] - out[["open", "close"]].max(axis=1)).clip(lower=0) / body
    out["momentum_filter_long"] = out["rsi_14"] < 35
    out["momentum_filter_short"] = out["rsi_14"] > 65
    return out


def run_feature_pipeline(config: dict) -> None:
    raw_dir = Path(config["project"]["raw_data_dir"])
    proc_dir = Path(config["project"]["processed_data_dir"])

    for exchange in config["symbols"].keys():
        ex_dir = raw_dir / exchange
        if not ex_dir.exists():
            continue
        for path in ex_dir.glob("*_1m.parquet"):
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            feat = build_features(df)
            out = proc_dir / exchange / path.name.replace("_1m.parquet", "_features_1m.parquet")
            save_parquet(feat, out)
            LOGGER.info("Wrote features -> %s", out)
