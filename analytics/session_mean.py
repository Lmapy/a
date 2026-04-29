"""Session-anchored typical-price mean — OHLC-only proxy for VWAP.

The harness has no real volume (Dukascopy candles carry tick count
but not signed flow), so a true volume-weighted average price is
impossible to compute honestly. The replacement we adopt is a
session-anchored TYPICAL PRICE mean:

    typical_price = (high + low + close) / 3
    session_mean[t] = mean(typical_price[t' for t' in session, t' <= t])

That's an unweighted running mean of typical price within each
session, anchored at the session open. It captures the same "fair
value drift" idea VWAP captures, without pretending we have
volume data we don't.

Two related quantities the executor needs:

  session_mean             current running mean (one value per H4 bar)
  session_atr              current 14-bar ATR (price units)
  atr_distance             (close - session_mean) / session_atr

The `atr_distance_from_session_mean` filter accepts trades only
when the current bar's distance from the running session mean is
within a configurable band (in ATR units). Anchored per session
so morning data isn't mixed with afternoon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _typical_price(h4: pd.DataFrame) -> np.ndarray:
    return (h4["high"].values + h4["low"].values + h4["close"].values) / 3.0


def _atr(h4: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = h4["high"].values, h4["low"].values, h4["close"].values
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n, min_periods=n).mean().values


def session_anchored_mean(h4: pd.DataFrame, *,
                           session_start_hour_utc: int = 13) -> np.ndarray:
    """Running typical-price mean restarted at each session open.

    For default 13:00 UTC start the mean resets at the first H4 bar
    on or after 13:00 each calendar day; bars before 13:00 are
    treated as belonging to the prior session (pre-open).
    """
    df = h4.sort_values("time").reset_index(drop=True)
    tp = _typical_price(df)
    # session id = (date) + 1 if hour >= start_hour else 0
    hour = df["time"].dt.hour.values
    date = df["time"].dt.normalize().values.astype("datetime64[D]").astype(int)
    sid = date * 2 + (hour >= session_start_hour_utc).astype(int)
    out = np.empty_like(tp, dtype=float)
    cum = 0.0
    n = 0
    last_sid = None
    for i in range(len(tp)):
        if sid[i] != last_sid:
            cum = 0.0
            n = 0
            last_sid = sid[i]
        cum += tp[i]
        n += 1
        out[i] = cum / n
    return out


def atr_distance_from_session_mean(h4: pd.DataFrame, *,
                                    atr_n: int = 14,
                                    session_start_hour_utc: int = 13,
                                    ) -> np.ndarray:
    """Distance from session mean in ATR units (signed).

    Negative = price below the session mean; positive = above.
    """
    s_mean = session_anchored_mean(
        h4, session_start_hour_utc=session_start_hour_utc)
    atr = _atr(h4, atr_n)
    c = h4["close"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        return (c - s_mean) / atr
