"""Session analysis: Asian range, session-specific logic."""

import pandas as pd
import numpy as np

from config import SESSIONS


def compute_asian_range(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute the Asian session high/low for each trading day.

    Returns DataFrame with columns: date, asian_high, asian_low.
    """
    if "session" not in candles.columns:
        raise ValueError("Candles must have 'session' column. Run tag_sessions first.")

    asian = candles[candles["session"] == "asian"].copy()
    if asian.empty:
        return pd.DataFrame(columns=["date", "asian_high", "asian_low"])

    # Get the trading date: Asian session starting at 6PM belongs to the NEXT calendar day
    if asian.index.tz is None:
        ct_idx = asian.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = asian.index.tz_convert("US/Central")

    # If hour >= 18, trading date is next day; otherwise it's current day
    dates = ct_idx.date
    hours = ct_idx.hour
    trading_dates = pd.to_datetime(pd.Series(dates, index=asian.index).astype(str))
    # For evening session (>= 18), shift date forward
    evening_mask = hours >= 18
    trading_dates.loc[evening_mask] = trading_dates.loc[evening_mask] + pd.Timedelta(days=1)

    asian["trading_date"] = trading_dates.dt.date.values

    result = asian.groupby("trading_date").agg(
        asian_high=("high", "max"),
        asian_low=("low", "min"),
    ).reset_index().rename(columns={"trading_date": "date"})

    return result


def get_previous_day_hl(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute previous day high/low for each trading day.

    Returns DataFrame with columns: date, prev_high, prev_low.
    """
    if candles.index.tz is None:
        ct_idx = candles.index.tz_localize("UTC").tz_convert("US/Central")
    else:
        ct_idx = candles.index.tz_convert("US/Central")

    candles_c = candles.copy()
    candles_c["trading_date"] = ct_idx.date

    daily = candles_c.groupby("trading_date").agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
    ).reset_index()

    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily = daily.dropna(subset=["prev_high"]).rename(columns={"trading_date": "date"})

    return daily[["date", "prev_high", "prev_low"]]


def is_tradeable_session(session: str) -> bool:
    """Only trade during London and NY sessions."""
    return session in ("london", "ny")
