from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
         session_starts: pd.Series | None = None) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()

    if session_starts is not None:
        # Reset cumulative sums at session boundaries
        groups = session_starts.cumsum()
        cum_tp_vol = (typical_price * volume).groupby(groups).cumsum()
        cum_vol = volume.groupby(groups).cumsum()

    return cum_tp_vol / cum_vol.replace(0, np.nan)


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0
                    ) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def bollinger_bandwidth(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    upper, middle, lower = bollinger_bands(close, period, std_dev)
    return ((upper - lower) / middle.replace(0, np.nan)) * 100


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(high, low, close, period)

    plus_di = 100 * ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return ema(dx, period)


def adx_with_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
                ) -> tuple[pd.Series, pd.Series, pd.Series]:
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(high, low, close, period)

    plus_di = 100 * ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = ema(dx, period)
    return adx_val, plus_di, minus_di


def compute_indicators(df: pd.DataFrame, session_starts: pd.Series | None = None) -> pd.DataFrame:
    """Compute all standard indicators on an OHLCV DataFrame.

    Expects columns: open, high, low, close, volume.
    Returns the same DataFrame with indicator columns added.
    """
    df = df.copy()
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    df["ema_20"] = ema(c, 20)
    df["ema_50"] = ema(c, 50)
    df["sma_20"] = sma(c, 20)
    df["atr_14"] = atr(h, l, c, 14)
    df["rsi_14"] = rsi(c, 14)
    df["adx_14"], df["plus_di"], df["minus_di"] = adx_with_di(h, l, c, 14)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger_bands(c, 20, 2.0)
    df["bb_width"] = bollinger_bandwidth(c, 20, 2.0)
    df["vwap"] = vwap(h, l, c, v, session_starts)
    df["volume_sma"] = sma(v.astype(float), 20)

    return df
