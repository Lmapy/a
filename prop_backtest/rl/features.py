"""Observation space feature engineering for the RL environment."""
from __future__ import annotations

import math

import numpy as np

from prop_backtest.account.state import AccountState
from prop_backtest.data.loader import BarData


def build_observation(
    history: list[BarData],
    account: AccountState,
    window: int = 20,
) -> np.ndarray:
    """Build a flat float32 observation vector.

    Layout (total = window*6 + 6 + 2 = window*6 + 8):
        price_features  [window × 5]   Normalised OHLCV for the last `window` bars.
        return_features [window × 1]   Close-to-close % returns.
        account_features [6]           Account state ratios.
        time_features   [2]            Sine/cosine of hour-of-day.

    Normalisation:
        Prices / mean_close so the network is scale-invariant across instruments.
        Account values / starting_balance.
        Volume / mean_volume (clipped to avoid outliers).
    """
    window_bars = history[-window:]
    n = len(window_bars)

    closes = np.array([b.close for b in window_bars], dtype=np.float32)
    mean_close = closes.mean()
    if mean_close == 0:
        mean_close = 1.0

    # ── Price features (normalised OHLCV) ─────────────────────────────────
    price_feats = np.zeros((window, 5), dtype=np.float32)
    volumes = np.array([float(b.volume) for b in window_bars], dtype=np.float32)
    mean_vol = volumes.mean()
    if mean_vol == 0:
        mean_vol = 1.0

    for i, bar in enumerate(window_bars):
        price_feats[i, 0] = bar.open / mean_close
        price_feats[i, 1] = bar.high / mean_close
        price_feats[i, 2] = bar.low / mean_close
        price_feats[i, 3] = bar.close / mean_close
        price_feats[i, 4] = min(bar.volume / mean_vol, 5.0)   # clip volume ratio

    # ── Return features ────────────────────────────────────────────────────
    ret_feats = np.zeros(window, dtype=np.float32)
    for i in range(1, n):
        prev_close = window_bars[i - 1].close
        if prev_close != 0:
            ret_feats[i] = (window_bars[i].close - prev_close) / prev_close

    # ── Account state features ─────────────────────────────────────────────
    start = account.starting_balance
    account_feats = np.array([
        account.equity / start,                              # [0] equity ratio
        account.open_pnl / start,                           # [1] open pnl ratio
        float(np.sign(account.position_contracts)),          # [2] position: -1, 0, +1
        np.clip(account.dd_floor_proximity, 0.0, 5.0),      # [3] safety buffer (clipped)
        account.current_day_pnl / start,                    # [4] today's pnl ratio
        (account.intraday_hwm - start) / start,             # [5] profit above start
    ], dtype=np.float32)

    # ── Time features (hour-of-day as cyclical encoding) ──────────────────
    latest = window_bars[-1].timestamp
    hour_frac = latest.hour + latest.minute / 60.0
    time_feats = np.array([
        math.sin(2 * math.pi * hour_frac / 24.0),
        math.cos(2 * math.pi * hour_frac / 24.0),
    ], dtype=np.float32)

    return np.concatenate([
        price_feats.flatten(),
        ret_feats,
        account_feats,
        time_feats,
    ])


def observation_size(window: int = 20) -> int:
    """Return the length of the observation vector for a given window."""
    return window * 5 + window + 6 + 2
