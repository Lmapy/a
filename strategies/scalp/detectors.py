"""CBR scalp detectors: expansion, sweep, rebalance, market-structure shift.

All detectors operate on already-closed bars. They never look ahead.
Each detector exposes a stateless `evaluate(df, end_idx, cfg) -> result`
function so the engine can call them at every minute bar without
rebuilding state.

Where rolling windows are needed (ATR, pivot confirmation), the
caller pre-computes those columns once and the detector reads
slices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from strategies.scalp.config import (
    ExpansionConfig, StructureConfig, TriggerConfig,
)


# ---- rolling indicators --------------------------------------------------

def attach_atr(df: pd.DataFrame, n: int = 14, col: str = "atr") -> pd.DataFrame:
    """Add rolling ATR(n) on the input frame. Uses Wilder/SMA convention
    on true range. The column is shifted by 0 here -- callers use
    `df[col].iloc[i-1]` for no-lookahead."""
    out = df.copy()
    h = out["high"].values
    l = out["low"].values
    c = out["close"].values
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    out[col] = pd.Series(tr).rolling(n, min_periods=n).mean().values
    return out


# ---- one-sided expansion -------------------------------------------------

@dataclass
class ExpansionResult:
    detected: bool
    direction: int                   # +1 bull, -1 bear, 0 none
    start_idx: int = -1
    end_idx: int = -1
    high: float = float("nan")
    low: float = float("nan")
    midpoint: float = float("nan")
    net_move: float = 0.0
    atr_at_end: float = float("nan")
    n_directional: int = 0
    quality_score: float = 0.0       # 0..100


def evaluate_expansion(df: pd.DataFrame, end_idx: int,
                        cfg: ExpansionConfig) -> ExpansionResult:
    """Detect a one-sided expansion ending at `end_idx` (inclusive).

    The lookback is `[end_idx - lookback + 1, end_idx]` (all bars
    already closed by the time the engine calls this for bar
    `end_idx + 1`).
    """
    lookback = cfg.expansion_lookback_bars
    start_idx = end_idx - lookback + 1
    if start_idx < 1 or end_idx >= len(df):
        return ExpansionResult(detected=False, direction=0)

    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["atr"].values

    atr_end = atr[end_idx]
    if not math.isfinite(atr_end) or atr_end <= 0:
        return ExpansionResult(detected=False, direction=0)

    sl = slice(start_idx, end_idx + 1)
    if cfg.expansion_measurement_mode == "CLOSE_TO_CLOSE":
        net_move = c[end_idx] - c[start_idx]
    elif cfg.expansion_measurement_mode == "BODY_ONLY":
        # sum of signed bodies
        net_move = float((c[sl] - o[sl]).sum())
    else:  # HIGH_LOW
        net_move = h[end_idx] - l[start_idx] if c[end_idx] >= c[start_idx] \
            else -(h[start_idx] - l[end_idx])

    direction = 1 if net_move > 0 else -1 if net_move < 0 else 0

    # min absolute net move
    if abs(net_move) < cfg.min_net_move_atr_multiple * atr_end:
        return ExpansionResult(detected=False, direction=direction,
                                start_idx=start_idx, end_idx=end_idx,
                                net_move=net_move, atr_at_end=atr_end)

    # directional candle ratio
    bullish = (c[sl] > o[sl]).sum()
    bearish = (c[sl] < o[sl]).sum()
    n_directional = int(bullish if direction > 0 else bearish)
    pct = n_directional / lookback
    if pct < cfg.min_directional_candle_percent:
        return ExpansionResult(detected=False, direction=direction,
                                start_idx=start_idx, end_idx=end_idx,
                                high=float(h[sl].max()),
                                low=float(l[sl].min()),
                                midpoint=float((h[sl].max() + l[sl].min()) / 2),
                                net_move=net_move,
                                n_directional=n_directional,
                                atr_at_end=atr_end)

    # min total range
    total_range = float(h[sl].max() - l[sl].min())
    if total_range < cfg.minimum_total_range_atr_multiple * atr_end:
        return ExpansionResult(detected=False, direction=direction,
                                start_idx=start_idx, end_idx=end_idx,
                                high=float(h[sl].max()),
                                low=float(l[sl].min()),
                                net_move=net_move,
                                atr_at_end=atr_end,
                                n_directional=n_directional)

    high_v = float(h[sl].max())
    low_v = float(l[sl].min())
    midpoint = (high_v + low_v) / 2.0

    # quality score: blend directional %, net-vs-range, and pullback share
    score_dir = min(1.0, pct / 0.85) * 40           # 0-40
    score_strength = min(1.0, abs(net_move) / (2.0 * atr_end)) * 35  # 0-35
    # cleanliness: how much of the total range was "with" the trend
    # vs against. Sum of |body| against direction divided by total.
    against_body = float(np.maximum(0.0, -np.sign(net_move) * (c[sl] - o[sl])).sum())
    total_body = float(np.abs(c[sl] - o[sl]).sum()) or 1e-9
    cleanliness = max(0.0, 1.0 - against_body / total_body)
    score_clean = cleanliness * 25                  # 0-25
    quality = round(min(100.0, score_dir + score_strength + score_clean), 1)

    return ExpansionResult(
        detected=True, direction=direction,
        start_idx=start_idx, end_idx=end_idx,
        high=high_v, low=low_v, midpoint=midpoint,
        net_move=net_move, atr_at_end=atr_end,
        n_directional=n_directional,
        quality_score=quality,
    )


# ---- sweep & rebalance ---------------------------------------------------

@dataclass
class SweepResult:
    detected: bool
    direction: int = 0                       # +1 bull, -1 bear
    swept_level: float = float("nan")
    sweep_idx: int = -1
    reclaim_idx: int = -1
    sweep_depth: float = 0.0


def evaluate_sweep(df: pd.DataFrame, end_idx: int,
                    prev_h1_high: float, prev_h1_low: float,
                    lookback: int = 30) -> SweepResult:
    """Detect a sweep + reclaim of the previous-completed-1H high/low
    that finishes at or before `end_idx`. The 1H levels are passed in
    as scalars (caller has the 1H frame and resolves the right
    previous bar)."""
    if end_idx < 1 or end_idx >= len(df) or lookback <= 0:
        return SweepResult(detected=False)
    if not (math.isfinite(prev_h1_high) and math.isfinite(prev_h1_low)):
        return SweepResult(detected=False)

    start = max(0, end_idx - lookback + 1)
    h = df["high"].values[start:end_idx + 1]
    l = df["low"].values[start:end_idx + 1]
    c = df["close"].values[start:end_idx + 1]

    # Bearish reversal: sweep above prev H1 high, close back below
    above = h > prev_h1_high
    if above.any():
        first_sweep = int(np.argmax(above))   # first True
        # reclaim: any bar AFTER first_sweep whose close is BACK below the level
        for j in range(first_sweep + 1, len(c)):
            if c[j] < prev_h1_high:
                return SweepResult(
                    detected=True, direction=-1,
                    swept_level=prev_h1_high,
                    sweep_idx=start + first_sweep,
                    reclaim_idx=start + j,
                    sweep_depth=float(h[first_sweep] - prev_h1_high),
                )

    # Bullish reversal: sweep below prev H1 low, close back above
    below = l < prev_h1_low
    if below.any():
        first_sweep = int(np.argmax(below))
        for j in range(first_sweep + 1, len(c)):
            if c[j] > prev_h1_low:
                return SweepResult(
                    detected=True, direction=1,
                    swept_level=prev_h1_low,
                    sweep_idx=start + first_sweep,
                    reclaim_idx=start + j,
                    sweep_depth=float(prev_h1_low - l[first_sweep]),
                )
    return SweepResult(detected=False)


@dataclass
class RebalanceResult:
    detected: bool
    direction: int = 0
    midpoint: float = float("nan")
    rebalance_idx: int = -1
    distance_from_midpoint: float = float("nan")
    midpoint_touched: bool = False


def evaluate_rebalance(df: pd.DataFrame, end_idx: int,
                        expansion: ExpansionResult,
                        cfg: TriggerConfig,
                        atr_at_end: float,
                        tick_size: float) -> RebalanceResult:
    """After a one-sided expansion, did price return to (or near) the
    expansion midpoint within `max_bars_after_expansion`?

    Bullish rebalance after bearish expansion: price comes back UP
    to the midpoint -> we expect a continuation back UP, so the
    rebalance signal is bullish.
    Bearish rebalance after bullish expansion: opposite.
    """
    if not expansion.detected:
        return RebalanceResult(detected=False)
    end_window = expansion.end_idx + cfg.max_bars_after_expansion
    if end_idx <= expansion.end_idx or end_idx > end_window:
        return RebalanceResult(detected=False)

    mid = expansion.midpoint
    if not math.isfinite(mid):
        return RebalanceResult(detected=False)

    # tolerance in price units: max(ticks, ATR fraction)
    tol = max(cfg.rebalance_tolerance_ticks * tick_size,
              cfg.rebalance_tolerance_atr_fraction * (atr_at_end if math.isfinite(atr_at_end) else 0))

    h = float(df["high"].iloc[end_idx])
    l = float(df["low"].iloc[end_idx])
    c = float(df["close"].iloc[end_idx])

    touched = (l - tol) <= mid <= (h + tol)
    if cfg.require_midpoint_touch and not touched:
        return RebalanceResult(detected=False, direction=-expansion.direction,
                                midpoint=mid,
                                distance_from_midpoint=min(abs(c - mid),
                                                            abs(h - mid),
                                                            abs(l - mid)))
    distance = min(abs(c - mid), abs(h - mid), abs(l - mid))
    return RebalanceResult(
        detected=touched if cfg.require_midpoint_touch else (distance <= tol),
        direction=-expansion.direction,    # rebalance reverses the expansion
        midpoint=mid,
        rebalance_idx=end_idx,
        distance_from_midpoint=distance,
        midpoint_touched=touched,
    )


# ---- confirmed pivots + market structure shift ---------------------------

@dataclass
class Pivot:
    idx: int                # bar index of the pivot extreme
    price: float
    direction: int          # +1 high, -1 low
    confirmed_at_idx: int   # idx after which the pivot is "confirmed"


def find_confirmed_pivots(df: pd.DataFrame, *,
                           pivot_left: int = 2,
                           pivot_right: int = 2,
                           up_to_idx: int | None = None) -> list[Pivot]:
    """Scan `df` (up to `up_to_idx` inclusive) for confirmed pivots.

    A pivot at index `k` is CONFIRMED at index `k + pivot_right` --
    i.e. only after we've seen `pivot_right` bars whose high (low)
    is below (above) the candidate pivot. This is the no-lookahead
    rule the user asked for.
    """
    if up_to_idx is None:
        up_to_idx = len(df) - 1
    h = df["high"].values
    l = df["low"].values
    out: list[Pivot] = []
    for k in range(pivot_left, up_to_idx - pivot_right + 1):
        is_high = (h[k] >= h[k - pivot_left:k + pivot_right + 1]).all() and \
                   (h[k] > h[k - 1]) and (h[k] > h[k + 1])
        is_low = (l[k] <= l[k - pivot_left:k + pivot_right + 1]).all() and \
                  (l[k] < l[k - 1]) and (l[k] < l[k + 1])
        if is_high:
            out.append(Pivot(idx=k, price=float(h[k]),
                              direction=1,
                              confirmed_at_idx=k + pivot_right))
        elif is_low:
            out.append(Pivot(idx=k, price=float(l[k]),
                              direction=-1,
                              confirmed_at_idx=k + pivot_right))
    return out


@dataclass
class MSBResult:
    detected: bool
    direction: int = 0                       # +1 bull MSB, -1 bear
    broken_pivot_idx: int = -1
    broken_pivot_price: float = float("nan")
    break_idx: int = -1
    break_price: float = float("nan")
    impulse_origin_idx: int = -1
    impulse_origin_price: float = float("nan")
    impulse_high_or_low_idx: int = -1
    impulse_high_or_low_price: float = float("nan")


def evaluate_msb(df: pd.DataFrame, end_idx: int,
                  pivots: list[Pivot],
                  cfg: StructureConfig,
                  trigger_idx: int,
                  trigger_direction: int,
                  max_bars_between: int) -> MSBResult:
    """Did price break the most-recent confirmed swing in the
    direction implied by `trigger_direction`?

    `trigger_idx` is the bar index of the most recent sweep / rebalance
    completion. We only look for an MSB whose `break_idx` is within
    [trigger_idx, trigger_idx + max_bars_between].
    """
    if trigger_direction not in (-1, 1):
        return MSBResult(detected=False)
    if end_idx <= trigger_idx or end_idx > trigger_idx + max_bars_between:
        return MSBResult(detected=False)

    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # which pivots are ELIGIBLE? confirmed by end_idx, occurred BEFORE break.
    elig = [p for p in pivots if p.confirmed_at_idx <= end_idx and p.idx < end_idx]
    if not elig:
        return MSBResult(detected=False)

    # bullish MSB: break above most-recent confirmed swing high
    if trigger_direction == 1:
        # most recent confirmed swing high BEFORE end_idx
        highs = [p for p in elig if p.direction == 1 and p.idx < end_idx]
        if not highs:
            return MSBResult(detected=False)
        last_high = max(highs, key=lambda p: p.idx)
        break_check = (c[end_idx] > last_high.price
                        if cfg.structure_break_mode == "CLOSE_THROUGH"
                        else h[end_idx] > last_high.price)
        if not break_check:
            return MSBResult(detected=False)
        # impulse origin: most-recent confirmed LOW BEFORE the broken high
        lows_before = [p for p in elig if p.direction == -1 and p.idx < last_high.idx]
        if lows_before:
            origin = max(lows_before, key=lambda p: p.idx)
        else:
            origin = Pivot(idx=last_high.idx,
                            price=float(l[last_high.idx]),
                            direction=-1,
                            confirmed_at_idx=last_high.idx)
        # impulse high so far -- the highest high between origin and end
        sl = slice(origin.idx, end_idx + 1)
        imp_hi_idx = origin.idx + int(np.argmax(h[sl]))
        return MSBResult(
            detected=True, direction=1,
            broken_pivot_idx=last_high.idx,
            broken_pivot_price=last_high.price,
            break_idx=end_idx,
            break_price=float(c[end_idx]),
            impulse_origin_idx=origin.idx,
            impulse_origin_price=origin.price,
            impulse_high_or_low_idx=imp_hi_idx,
            impulse_high_or_low_price=float(h[imp_hi_idx]),
        )

    # bearish MSB: break below most-recent confirmed swing low
    lows = [p for p in elig if p.direction == -1 and p.idx < end_idx]
    if not lows:
        return MSBResult(detected=False)
    last_low = max(lows, key=lambda p: p.idx)
    break_check = (c[end_idx] < last_low.price
                    if cfg.structure_break_mode == "CLOSE_THROUGH"
                    else l[end_idx] < last_low.price)
    if not break_check:
        return MSBResult(detected=False)
    highs_before = [p for p in elig if p.direction == 1 and p.idx < last_low.idx]
    if highs_before:
        origin = max(highs_before, key=lambda p: p.idx)
    else:
        origin = Pivot(idx=last_low.idx, price=float(h[last_low.idx]),
                        direction=1, confirmed_at_idx=last_low.idx)
    sl = slice(origin.idx, end_idx + 1)
    imp_lo_idx = origin.idx + int(np.argmin(l[sl]))
    return MSBResult(
        detected=True, direction=-1,
        broken_pivot_idx=last_low.idx,
        broken_pivot_price=last_low.price,
        break_idx=end_idx,
        break_price=float(c[end_idx]),
        impulse_origin_idx=origin.idx,
        impulse_origin_price=origin.price,
        impulse_high_or_low_idx=imp_lo_idx,
        impulse_high_or_low_price=float(l[imp_lo_idx]),
    )
