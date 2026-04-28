"""TPO / Market Profile from OHLC candles only.

A TPO (Time Price Opportunity) profile partitions a session's price
range into discrete price bins, then counts how many TIME BRACKETS
(not how much volume) traded at each bin. The result is a histogram
of *time at price*. POC = bin with highest TPO count, value area =
the contiguous range of bins that contains a target share of total
TPOs (usually 70%), VAH = top of value area, VAL = bottom.

Why this is OHLC-only legitimate
--------------------------------
TPO is computed from the (high, low) range of each bracket — the
question is "did this bracket touch this price?", not "how much was
traded at this price?". Real volume is not required, and real volume
would not change a TPO profile.

This implementation does NOT pretend to be Volume Profile. The
returned object's field names use `tpo_*` not `volume_*`. The Batch
F feature-capability registry rejects any token containing
`volume_profile` or `volume_poc` etc.; TPO tokens (`tpo_poc`,
`tpo_vah`, `tpo_val`, ...) are explicitly allowed.

Inputs / outputs
----------------
`compute_tpo_profile(candles, bin_size, bracket_minutes,
session_start, session_end, value_area_pct=0.70)` ->
`TPOProfile` dataclass with:

  * poc           price at peak TPO count
  * vah / val     top / bottom of the value area
  * value_area_width
  * tpo_counts    {price -> count}
  * single_prints sorted list of prices touched in exactly one bracket
  * poor_high     bool (multiple touches of the high; range edge is
                  poorly tested -> typical re-test setup)
  * poor_low      bool (same on the low side)
  * excess_high   bool (single-print at top — signals exhaustion)
  * excess_low    bool (single-print at bottom)
  * initial_balance_high / _low  (range of the first
                                  `initial_balance_brackets` brackets)
  * open_inside_value, open_outside_value  if a `prior_value_area`
                                            tuple is given

The candle dataframe must have columns:
    time (tz-aware UTC), high, low,   open, close (open/close optional)

Bracket assignment
------------------
Every candle is assigned to a bracket by integer-flooring its time-
since-session-start to `bracket_minutes`. We then take the
[bracket_low, bracket_high] across all candles in the bracket.

If you pass M15 candles with `bracket_minutes=30`, two M15 candles
form one bracket. If you pass M30 candles with `bracket_minutes=30`,
each candle is its own bracket. Either way the TPO count is the
number of brackets whose range overlaps the bin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class TPOProfile:
    poc: float
    vah: float
    val: float
    value_area_width: float
    tpo_counts: dict[float, int]            # price -> bracket count
    n_brackets: int
    bin_size: float
    single_prints: list[float]
    poor_high: bool
    poor_low: bool
    excess_high: bool
    excess_low: bool
    initial_balance_high: float | None = None
    initial_balance_low: float | None = None
    open_inside_value: bool | None = None
    open_outside_value: bool | None = None
    session_open: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    session_close: float | None = None
    bracket_minutes: int = 30
    initial_balance_brackets: int = 2

    def to_json(self) -> dict:
        return {
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val,
            "value_area_width": self.value_area_width,
            "n_brackets": self.n_brackets,
            "bin_size": self.bin_size,
            "single_prints": self.single_prints,
            "poor_high": self.poor_high,
            "poor_low": self.poor_low,
            "excess_high": self.excess_high,
            "excess_low": self.excess_low,
            "initial_balance_high": self.initial_balance_high,
            "initial_balance_low": self.initial_balance_low,
            "open_inside_value": self.open_inside_value,
            "open_outside_value": self.open_outside_value,
            "session_open": self.session_open,
            "session_high": self.session_high,
            "session_low": self.session_low,
            "session_close": self.session_close,
            "bracket_minutes": self.bracket_minutes,
            "initial_balance_brackets": self.initial_balance_brackets,
        }


# --- helpers -----------------------------------------------------------------

def _floor_to_bin(price: float, bin_size: float) -> float:
    """Floor a price down to the bin grid. Uses round() to avoid float
    drift on simple inputs like 2000.5 / 0.5."""
    return round(math.floor(price / bin_size) * bin_size, 8)


def _bracket_range(candles: pd.DataFrame, bin_size: float) -> tuple[float, float]:
    return (_floor_to_bin(float(candles["low"].min()), bin_size),
            _floor_to_bin(float(candles["high"].max()), bin_size))


def _value_area(counts: dict[float, int], target_share: float) -> tuple[float, float, float]:
    """Expand from the POC outward until `target_share` of the total
    bracket count is captured. Standard market-profile convention.
    Ties broken by preferring the higher side first (matches CME doc)."""
    if not counts:
        return float("nan"), float("nan"), float("nan")
    total = sum(counts.values())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    target = target_share * total
    prices = sorted(counts.keys())
    poc = max(counts, key=lambda p: (counts[p], p))
    poc_idx = prices.index(poc)
    captured = counts[poc]
    lo_idx = hi_idx = poc_idx
    while captured < target and (lo_idx > 0 or hi_idx < len(prices) - 1):
        # Look at the next candidates above and below; take whichever
        # contributes more, tie-break upward.
        up = counts[prices[hi_idx + 1]] if hi_idx + 1 < len(prices) else -1
        dn = counts[prices[lo_idx - 1]] if lo_idx - 1 >= 0 else -1
        if up >= dn and hi_idx + 1 < len(prices):
            hi_idx += 1
            captured += counts[prices[hi_idx]]
        elif lo_idx - 1 >= 0:
            lo_idx -= 1
            captured += counts[prices[lo_idx]]
        else:
            break
    val = prices[lo_idx]
    vah = prices[hi_idx]
    return poc, vah, val


# --- public API --------------------------------------------------------------

def compute_tpo_profile(candles: pd.DataFrame,
                         *,
                         bin_size: float = 0.50,
                         bracket_minutes: int = 30,
                         value_area_pct: float = 0.70,
                         initial_balance_brackets: int = 2,
                         prior_value_area: tuple[float, float] | None = None,
                         ) -> TPOProfile:
    """Build a TPO profile from a session's worth of OHLC candles.

    `candles` must be filtered to a single session by the caller (this
    keeps the function simple and explicit; compute one profile per
    session, not per chained day). It must be sorted by `time`.

    `bin_size` is in price units. For XAUUSD at $2000/oz, 0.50 is a
    reasonable default. `bracket_minutes` is the TPO bracket length;
    30 min is the conventional choice.
    """
    if candles.empty:
        return TPOProfile(
            poc=float("nan"), vah=float("nan"), val=float("nan"),
            value_area_width=float("nan"), tpo_counts={}, n_brackets=0,
            bin_size=bin_size, single_prints=[], poor_high=False,
            poor_low=False, excess_high=False, excess_low=False,
            bracket_minutes=bracket_minutes,
            initial_balance_brackets=initial_balance_brackets,
        )
    df = candles.sort_values("time").reset_index(drop=True)
    sess_start = df["time"].iloc[0]
    # bracket id = integer floor(minutes-since-session-start / bracket_minutes)
    minutes_since = ((df["time"] - sess_start).dt.total_seconds() / 60.0).astype(int)
    df = df.assign(_bracket=(minutes_since // bracket_minutes))

    # bracket -> [low, high]
    counts: dict[float, int] = {}
    bracket_ranges: list[tuple[int, float, float]] = []
    for bid, g in df.groupby("_bracket"):
        b_lo, b_hi = _bracket_range(g, bin_size)
        bracket_ranges.append((int(bid), b_lo, b_hi))
    n_brackets = len(bracket_ranges)
    if n_brackets == 0:
        return TPOProfile(
            poc=float("nan"), vah=float("nan"), val=float("nan"),
            value_area_width=float("nan"), tpo_counts={}, n_brackets=0,
            bin_size=bin_size, single_prints=[], poor_high=False,
            poor_low=False, excess_high=False, excess_low=False,
            bracket_minutes=bracket_minutes,
            initial_balance_brackets=initial_balance_brackets,
        )

    # bin counts
    for _, b_lo, b_hi in bracket_ranges:
        # iterate bins from b_lo to b_hi inclusive
        n_bins = int(round((b_hi - b_lo) / bin_size)) + 1
        for k in range(n_bins):
            p = round(b_lo + k * bin_size, 8)
            counts[p] = counts.get(p, 0) + 1

    poc, vah, val = _value_area(counts, value_area_pct)

    # session aggregates
    session_high = float(df["high"].max())
    session_low = float(df["low"].min())
    session_open = float(df["open"].iloc[0]) if "open" in df.columns else None
    session_close = float(df["close"].iloc[-1]) if "close" in df.columns else None

    # single prints: prices touched in only one bracket
    single_prints = sorted(p for p, c in counts.items() if c == 1)

    # poor highs/lows: top/bottom price has > 1 bracket count -> the
    # range edge has been "tested" multiple times (poor edge in
    # auction-theory terms; typical re-test setup).
    top = max(counts.keys())
    bot = min(counts.keys())
    poor_high = counts.get(top, 0) > 1
    poor_low = counts.get(bot, 0) > 1
    # excess: the very top or very bottom is a single print
    excess_high = counts.get(top, 0) == 1
    excess_low = counts.get(bot, 0) == 1

    # Initial balance: range of the first `initial_balance_brackets` brackets.
    ib_brackets = sorted({int(bid) for bid, _, _ in bracket_ranges})[:initial_balance_brackets]
    if ib_brackets:
        ib_mask = df["_bracket"].isin(ib_brackets)
        ib_high = float(df.loc[ib_mask, "high"].max())
        ib_low = float(df.loc[ib_mask, "low"].min())
    else:
        ib_high = ib_low = None

    # Open inside / outside prior value
    open_inside_value = open_outside_value = None
    if prior_value_area is not None and session_open is not None:
        prev_val, prev_vah = prior_value_area
        open_inside_value = bool(prev_val <= session_open <= prev_vah)
        open_outside_value = not open_inside_value

    value_width = (vah - val) if math.isfinite(vah) and math.isfinite(val) else float("nan")
    return TPOProfile(
        poc=poc, vah=vah, val=val,
        value_area_width=value_width,
        tpo_counts=counts, n_brackets=n_brackets,
        bin_size=bin_size,
        single_prints=single_prints,
        poor_high=poor_high, poor_low=poor_low,
        excess_high=excess_high, excess_low=excess_low,
        initial_balance_high=ib_high, initial_balance_low=ib_low,
        open_inside_value=open_inside_value,
        open_outside_value=open_outside_value,
        session_open=session_open, session_high=session_high,
        session_low=session_low, session_close=session_close,
        bracket_minutes=bracket_minutes,
        initial_balance_brackets=initial_balance_brackets,
    )


def session_slices(candles: pd.DataFrame, *,
                   session_start_hour_utc: int = 13,
                   session_length_hours: int = 7) -> Iterable[pd.DataFrame]:
    """Yield per-day session slices of `candles` clipped to
    [start_hour, start_hour + length) in UTC. Default = the regular
    NY session ~ 13:00-20:00 UTC (08:30-16:00 ET, no DST handling).
    Useful for batching `compute_tpo_profile` across history."""
    if candles.empty:
        return
    df = candles.sort_values("time").reset_index(drop=True)
    df = df.assign(
        _hour=df["time"].dt.hour,
        _date=df["time"].dt.normalize(),
    )
    end_hour = (session_start_hour_utc + session_length_hours) % 24
    if end_hour <= session_start_hour_utc:
        # session crosses midnight; slice each day on the open and the
        # next day on the close. For default 13:00 UTC + 7h this branch
        # is unused.
        in_sess = (df["_hour"] >= session_start_hour_utc) | (df["_hour"] < end_hour)
    else:
        in_sess = (df["_hour"] >= session_start_hour_utc) & (df["_hour"] < end_hour)
    for d, g in df[in_sess].groupby("_date"):
        yield g.drop(columns=["_hour", "_date"]).reset_index(drop=True)
