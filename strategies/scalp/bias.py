"""1H higher-timeframe bias + optional DXY filter.

Both modules return a `Bias` enum-like int per minute timestamp:
  +1 = bullish, -1 = bearish, 0 = neutral / unknown.

The H1 bias resolver uses ONLY completed 1H bars: at minute time
`t`, the relevant 1H bar is the one whose end is <= t. That excludes
the still-forming 1H bar.

The DXY filter degrades gracefully: if the DXY frame is None or empty
the filter returns +1 / -1 only when the gold side asks for it (i.e.
"agree" = always pass), and the engine logs `dxy_available=False`
in the setup row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.scalp.config import DXYConfig, HTFBiasConfig


@dataclass
class BiasSnapshot:
    direction: int            # +1 / -1 / 0
    reason: str               # short label for the setup log


# ---- 1H bias --------------------------------------------------------------

def precompute_h1_bias(h1: pd.DataFrame, cfg: HTFBiasConfig) -> pd.DataFrame:
    """Add a `bias` column to a copy of the H1 frame, computed using
    each H1 bar's own data. The engine resolves "previous COMPLETED
    H1 at minute t" by index-shifting this column.
    """
    out = h1.sort_values("time").reset_index(drop=True).copy()
    if cfg.bias_mode == "OFF":
        out["bias"] = 0
        out["bias_reason"] = "off"
        return out

    if cfg.bias_mode == "PREVIOUS_1H_CANDLE_DIRECTION":
        col = np.sign(out["close"].values - out["open"].values).astype(int)
        out["bias"] = col
        out["bias_reason"] = np.where(col > 0, "h1_green",
                              np.where(col < 0, "h1_red", "h1_doji"))
        return out

    if cfg.bias_mode == "EMA_SLOPE":
        c = out["close"].values
        ema = pd.Series(c).ewm(span=cfg.ema_length, adjust=False).mean().values
        ema_prev = np.concatenate(([np.nan], ema[:-1]))
        rising = ema > ema_prev
        bias = np.where((c > ema) & rising, 1,
                np.where((c < ema) & ~rising & np.isfinite(ema_prev), -1, 0))
        out["bias"] = bias
        out["bias_reason"] = np.where(bias > 0, "ema_up",
                              np.where(bias < 0, "ema_dn", "ema_flat"))
        return out

    if cfg.bias_mode == "WICK_BODY_BIAS":
        h = out["high"].values
        l = out["low"].values
        o = out["open"].values
        c = out["close"].values
        rng = np.maximum(h - l, 1e-12)
        body_high = np.maximum(o, c)
        body_low = np.minimum(o, c)
        upper_wick = h - body_high
        lower_wick = body_low - l
        mid = (h + l) / 2.0
        bull_wick = (lower_wick / rng) >= cfg.wick_body_threshold
        bear_wick = (upper_wick / rng) >= cfg.wick_body_threshold
        bias = np.where(bull_wick & (c > mid), 1,
                np.where(bear_wick & (c < mid), -1, 0))
        out["bias"] = bias
        out["bias_reason"] = np.where(bias > 0, "lower_wick_close_above_mid",
                              np.where(bias < 0, "upper_wick_close_below_mid",
                                        "no_wick_bias"))
        return out

    raise ValueError(f"unknown bias_mode: {cfg.bias_mode}")


def resolve_h1_bias_at(minute_ts: pd.Timestamp,
                        h1: pd.DataFrame) -> BiasSnapshot:
    """Find the most-recent H1 bar whose end is <= `minute_ts`. The H1
    frame's `time` column holds bar OPEN times; bar END = open + 1 hour.
    No-lookahead: never use a bar whose end is > minute_ts.
    """
    if h1.empty:
        return BiasSnapshot(0, "no_h1_data")
    end_times = h1["time"] + pd.Timedelta(hours=1)
    mask = end_times <= minute_ts
    if not mask.any():
        return BiasSnapshot(0, "h1_warmup")
    last_idx = int(mask.values.nonzero()[0][-1])
    direction = int(h1["bias"].iloc[last_idx])
    reason = str(h1["bias_reason"].iloc[last_idx])
    return BiasSnapshot(direction, reason)


# ---- DXY filter -----------------------------------------------------------

def precompute_dxy_bias(dxy: pd.DataFrame | None,
                         cfg: DXYConfig) -> pd.DataFrame | None:
    """Same shape as `precompute_h1_bias` but for the DXY frame.
    Returns None if dxy is None/empty or mode is OFF."""
    if cfg.dxy_mode == "OFF" or dxy is None or dxy.empty:
        return None
    out = dxy.sort_values("time").reset_index(drop=True).copy()
    if cfg.dxy_mode == "PREVIOUS_CANDLE_DIRECTION":
        col = np.sign(out["close"].values - out["open"].values).astype(int)
        out["dxy_bias"] = col
    elif cfg.dxy_mode == "EMA_SLOPE":
        c = out["close"].values
        ema = pd.Series(c).ewm(span=cfg.ema_length, adjust=False).mean().values
        ema_prev = np.concatenate(([np.nan], ema[:-1]))
        rising = ema > ema_prev
        out["dxy_bias"] = np.where((c > ema) & rising, 1,
                            np.where((c < ema) & ~rising & np.isfinite(ema_prev), -1, 0))
    elif cfg.dxy_mode == "CLOSE_VS_EMA":
        c = out["close"].values
        ema = pd.Series(c).ewm(span=cfg.ema_length, adjust=False).mean().values
        out["dxy_bias"] = np.sign(c - ema).astype(int)
    else:
        raise ValueError(f"unknown dxy_mode: {cfg.dxy_mode}")
    return out


def dxy_inverse_check(direction_gold: int,
                       minute_ts: pd.Timestamp,
                       dxy: pd.DataFrame | None,
                       cfg: DXYConfig) -> tuple[bool, str]:
    """Returns (passes, reason). For long gold setups DXY should be
    weak/bearish; for short gold setups DXY should be strong/bullish.

    If DXY data isn't available or the mode is OFF the filter passes
    automatically with reason `dxy_off` / `dxy_unavailable`."""
    if cfg.dxy_mode == "OFF":
        return True, "dxy_off"
    if dxy is None or dxy.empty or "dxy_bias" not in dxy.columns:
        return True, "dxy_unavailable"
    end_times = dxy["time"] + pd.Timedelta(hours=1)
    mask = end_times <= minute_ts
    if not mask.any():
        return True, "dxy_warmup"
    last_idx = int(mask.values.nonzero()[0][-1])
    dxy_dir = int(dxy["dxy_bias"].iloc[last_idx])
    if not cfg.require_inverse_confirmation:
        return True, "dxy_not_required"
    if direction_gold > 0:
        ok = dxy_dir <= 0      # gold long wants DXY bearish or neutral
    elif direction_gold < 0:
        ok = dxy_dir >= 0      # gold short wants DXY bullish or neutral
    else:
        return True, "no_gold_direction"
    return ok, ("dxy_inverse_ok" if ok else "dxy_aligned_against_gold")
