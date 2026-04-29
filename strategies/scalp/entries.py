"""Entry, stop, and target computation for the CBR scalp engine.

All math is pure (no I/O). The engine calls `compute_entry_plan(...)`
once an MSB has fired with the impulse origin / impulse extreme; the
plan is then placed as a limit (or market) order and the engine
walks subsequent bars to detect fill / stop / target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from strategies.scalp.config import (
    CBRGoldScalpConfig, EntryConfig, RiskConfig, StopTargetConfig,
)
from strategies.scalp.detectors import ExpansionResult, MSBResult


@dataclass
class EntryPlan:
    direction: int               # +1 long, -1 short
    entry_price: float
    stop_price: float
    target_price: float
    risk_per_unit: float         # |entry - stop| in price units
    reward_per_unit: float
    r_multiple: float            # reward / risk
    quantity: float              # contracts / units
    fill_kind: str               # "limit" or "market"
    expiry_bar_idx: int          # absolute index after which to cancel
    notes: str = ""


def _stop_price(direction: int, msb: MSBResult, exp: ExpansionResult,
                  st: StopTargetConfig, atr_at_entry: float,
                  tick_size: float) -> float:
    if st.stop_mode == "RECENT_SWING":
        # long: below origin low; short: above origin high
        if direction > 0:
            return msb.impulse_origin_price - st.stop_buffer_ticks * tick_size
        return msb.impulse_origin_price + st.stop_buffer_ticks * tick_size
    if st.stop_mode == "EXPANSION_EXTREME":
        if direction > 0:
            return exp.low - st.stop_buffer_ticks * tick_size
        return exp.high + st.stop_buffer_ticks * tick_size
    if st.stop_mode == "ATR":
        if not math.isfinite(atr_at_entry) or atr_at_entry <= 0:
            atr_at_entry = tick_size * 50
        if direction > 0:
            return msb.impulse_origin_price - st.atr_stop_multiple * atr_at_entry
        return msb.impulse_origin_price + st.atr_stop_multiple * atr_at_entry
    if st.stop_mode == "CUSTOM_TICKS":
        offset = st.custom_stop_ticks * tick_size
        if direction > 0:
            return msb.impulse_origin_price - offset
        return msb.impulse_origin_price + offset
    raise ValueError(f"unknown stop_mode: {st.stop_mode}")


def _target_price(direction: int, entry: float, stop: float,
                    msb: MSBResult, exp: ExpansionResult,
                    prev_h1_open: float, prev_h1_close: float,
                    asia_high: float, asia_low: float,
                    st: StopTargetConfig) -> float:
    risk = abs(entry - stop)
    if st.target_mode == "FIXED_R_MULTIPLE":
        return entry + direction * st.risk_reward * risk
    if st.target_mode == "EXPANSION_MIDPOINT":
        # for longs (rebalance back from below), target moves toward expansion mid
        return exp.midpoint
    if st.target_mode == "PREVIOUS_1H_EQUILIBRIUM":
        if math.isfinite(prev_h1_open) and math.isfinite(prev_h1_close):
            return (prev_h1_open + prev_h1_close) / 2
        return entry + direction * st.risk_reward * risk
    if st.target_mode == "OPPOSING_EXPANSION_EXTREME":
        return exp.high if direction > 0 else exp.low
    if st.target_mode == "SESSION_HIGH_LOW":
        if direction > 0 and math.isfinite(asia_high):
            return asia_high
        if direction < 0 and math.isfinite(asia_low):
            return asia_low
        return entry + direction * st.risk_reward * risk
    raise ValueError(f"unknown target_mode: {st.target_mode}")


def _entry_price(direction: int, msb: MSBResult,
                  cfg: EntryConfig, market_close_price: float) -> tuple[float, str]:
    impulse_low = msb.impulse_origin_price if direction > 0 \
                    else msb.impulse_high_or_low_price
    impulse_high = msb.impulse_high_or_low_price if direction > 0 \
                    else msb.impulse_origin_price
    body = impulse_high - impulse_low
    if cfg.entry_mode == "MARKET_ON_MSB_CLOSE":
        return market_close_price, "market"
    if cfg.entry_mode == "LIMIT_50_RETRACE":
        return (impulse_low + 0.5 * body), "limit"
    if cfg.entry_mode == "LIMIT_618_RETRACE":
        # 61.8% RETRACEMENT from the impulse extreme back toward origin
        # for long: from impulse_high down 61.8% of body
        return (impulse_low + (1 - 0.618) * body) if direction > 0 \
            else (impulse_high - (1 - 0.618) * body), "limit"
    if cfg.entry_mode == "LIMIT_CUSTOM_RETRACE":
        f = cfg.custom_retrace
        return (impulse_low + (1 - f) * body) if direction > 0 \
            else (impulse_high - (1 - f) * body), "limit"
    raise ValueError(f"unknown entry_mode: {cfg.entry_mode}")


def _qty(risk_price: float, cfg_risk: RiskConfig, equity: float) -> float:
    if cfg_risk.position_size_mode == "FIXED_QTY":
        return cfg_risk.fixed_qty
    if cfg_risk.position_size_mode == "FIXED_RISK_CURRENCY":
        per_unit_risk_currency = risk_price * (cfg_risk.tick_value / max(cfg_risk.tick_size, 1e-9))
        if per_unit_risk_currency <= 0:
            return 0.0
        return cfg_risk.fixed_risk_currency / per_unit_risk_currency
    if cfg_risk.position_size_mode == "PERCENT_EQUITY":
        target = equity * cfg_risk.risk_percent / 100.0
        per_unit_risk_currency = risk_price * (cfg_risk.tick_value / max(cfg_risk.tick_size, 1e-9))
        if per_unit_risk_currency <= 0:
            return 0.0
        return target / per_unit_risk_currency
    raise ValueError(f"unknown position_size_mode: {cfg_risk.position_size_mode}")


def compute_entry_plan(*,
                        cfg: CBRGoldScalpConfig,
                        msb: MSBResult,
                        expansion: ExpansionResult,
                        prev_h1_open: float, prev_h1_close: float,
                        asia_high: float, asia_low: float,
                        atr_at_entry: float,
                        market_close_price: float,
                        msb_idx: int,
                        equity: float) -> EntryPlan | None:
    """Returns a complete entry plan or None if the plan would be
    invalid (zero qty, distance constraints, etc.).
    """
    direction = msb.direction
    entry_price, fill_kind = _entry_price(direction, msb,
                                            cfg.entry, market_close_price)
    stop_price = _stop_price(direction, msb, expansion, cfg.stop_target,
                              atr_at_entry, cfg.risk.tick_size)

    # Sanity: stop must be on the expected side of entry
    if direction > 0 and stop_price >= entry_price:
        return None
    if direction < 0 and stop_price <= entry_price:
        return None

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        return None

    target_price = _target_price(direction, entry_price, stop_price,
                                   msb, expansion,
                                   prev_h1_open, prev_h1_close,
                                   asia_high, asia_low,
                                   cfg.stop_target)
    reward_per_unit = (target_price - entry_price) * direction
    if reward_per_unit <= 0:
        # target on wrong side of entry
        return None
    r = reward_per_unit / risk_per_unit

    # entry-distance gates
    distance_ticks = abs(entry_price - market_close_price) / cfg.risk.tick_size
    if distance_ticks < cfg.entry.min_entry_distance_ticks:
        return None
    if distance_ticks > cfg.entry.max_entry_distance_ticks:
        return None

    qty = _qty(risk_per_unit, cfg.risk, equity)
    if qty <= 0:
        return None

    return EntryPlan(
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_per_unit=risk_per_unit,
        reward_per_unit=reward_per_unit,
        r_multiple=r,
        quantity=qty,
        fill_kind=fill_kind,
        expiry_bar_idx=msb_idx + cfg.entry.entry_expiry_bars,
    )
