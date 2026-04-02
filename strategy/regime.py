"""Market regime detection: trending vs ranging filter.

Uses ADX (Average Directional Index) and price position relative to
moving averages to classify the current market regime.
"""

import numpy as np
import pandas as pd


def compute_adx(highs, lows, closes, period=14):
    """Compute ADX (Average Directional Index) for trend strength.

    ADX > 25: trending market (good for our pullback strategy)
    ADX < 20: ranging/choppy (avoid trading)
    """
    n = len(highs)
    adx = np.full(n, np.nan)

    if n < period * 2:
        return adx

    # True Range
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

    # Smoothed TR, +DM, -DM using Wilder's smoothing
    smoothed_tr = np.zeros(n)
    smoothed_plus = np.zeros(n)
    smoothed_minus = np.zeros(n)

    smoothed_tr[period] = np.sum(tr[1:period + 1])
    smoothed_plus[period] = np.sum(plus_dm[1:period + 1])
    smoothed_minus[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] / period + tr[i]
        smoothed_plus[i] = smoothed_plus[i - 1] - smoothed_plus[i - 1] / period + plus_dm[i]
        smoothed_minus[i] = smoothed_minus[i - 1] - smoothed_minus[i - 1] / period + minus_dm[i]

    # +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100 * smoothed_plus[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus[i] / smoothed_tr[i]

        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX: smoothed DX
    adx[period * 2] = np.mean(dx[period:period * 2 + 1])
    for i in range(period * 2 + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def compute_ema(data, period):
    """Compute Exponential Moving Average."""
    n = len(data)
    ema = np.full(n, np.nan)

    if n < period:
        return ema

    ema[period - 1] = np.mean(data[:period])
    multiplier = 2.0 / (period + 1)

    for i in range(period, n):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

    return ema


def classify_regime(highs, lows, closes, adx_period=14, ema_period=50):
    """Classify market regime for each bar.

    Returns array of regime labels:
    - 'trending_up': ADX > 22 and price above EMA (bullish trend)
    - 'trending_down': ADX > 22 and price below EMA (bearish trend)
    - 'ranging': ADX < 22 (choppy, avoid)
    - 'unknown': insufficient data
    """
    n = len(closes)
    regimes = np.full(n, 'unknown', dtype=object)

    adx = compute_adx(highs, lows, closes, period=adx_period)
    ema = compute_ema(closes, period=ema_period)

    for i in range(n):
        if np.isnan(adx[i]) or np.isnan(ema[i]):
            continue

        if adx[i] >= 18:
            if closes[i] > ema[i]:
                regimes[i] = 'trending_up'
            else:
                regimes[i] = 'trending_down'
        else:
            regimes[i] = 'ranging'

    return regimes, adx, ema
