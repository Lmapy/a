"""CBR scalp backtest engine.

Per-minute bar loop with an explicit per-session state machine.
Calls the detector / bias / entry modules; never reads forward.

State machine (per session):

    IDLE
      v   (expansion detected at end of bar i)
    EXPANSION_PENDING
      v   (sweep or rebalance detected within max_bars_after_expansion)
    TRIGGER_PENDING
      v   (MSB confirmed within max_bars_between_trigger_and_msb)
    ENTRY_PLACED   <-- limit order live
      v   (filled OR expired OR session closed)
    IN_TRADE       <-- stop / target walk
      v   (exit hit OR session closed)
    DONE   (or back to IDLE if `one_trade_per_session` is False)

Every state transition produces a setup-log row. Every entry, fill,
and exit produces a trade row.
"""
from __future__ import annotations

import datetime as _dt
import math
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategies.scalp.bias import (
    dxy_inverse_check, precompute_dxy_bias, precompute_h1_bias,
    resolve_h1_bias_at,
)
from strategies.scalp.config import CBRGoldScalpConfig
from strategies.scalp.detectors import (
    ExpansionResult, MSBResult, Pivot, RebalanceResult, SweepResult,
    attach_atr, evaluate_expansion, evaluate_msb, evaluate_rebalance,
    evaluate_sweep, find_confirmed_pivots,
)
from strategies.scalp.entries import EntryPlan, compute_entry_plan
from strategies.scalp.sessions import (
    SessionWindow, annotate_session_columns, asia_session_high_low,
)


# ---- per-session machinery -------------------------------------------------

@dataclass
class SessionState:
    """All the rolling state for ONE session day."""
    session_date: _dt.date
    expansion: ExpansionResult | None = None
    trigger_idx: int | None = None
    trigger_kind: str | None = None     # "sweep" | "rebalance"
    trigger_direction: int = 0
    sweep: SweepResult | None = None
    rebalance: RebalanceResult | None = None
    msb: MSBResult | None = None
    plan: EntryPlan | None = None
    in_trade: bool = False
    trade_entry_idx: int | None = None
    trade_entry_price: float | None = None
    trades_taken_today: int = 0
    setup_count: int = 0
    state: str = "IDLE"


@dataclass
class TradeRow:
    trade_id: int
    setup_id: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str
    r_result: float
    pnl: float
    duration_minutes: int
    mae: float
    mfe: float
    session_date: _dt.date
    day_of_week: str
    htf_bias: int
    dxy_state: str
    expansion_quality: float
    trigger_kind: str
    structure_mode: str
    retracement_level: float

    def to_dict(self) -> dict:
        return {**self.__dict__,
                "entry_time": self.entry_time.isoformat(),
                "exit_time": self.exit_time.isoformat(),
                "session_date": str(self.session_date)}


@dataclass
class SetupRow:
    setup_id: str
    timestamp: pd.Timestamp
    session_date: _dt.date
    symbol: str
    direction: int
    htf_bias: int
    htf_bias_reason: str
    dxy_state: str
    expansion_direction: int
    expansion_quality: float
    expansion_start: pd.Timestamp | None
    expansion_end: pd.Timestamp | None
    expansion_high: float
    expansion_low: float
    expansion_mid: float
    sweep_detected: bool
    swept_level: float
    rebalance_detected: bool
    rebalance_midpoint: float
    msb_detected: bool
    msb_break_level: float
    entry_mode: str
    entry_price: float
    stop_price: float
    target_price: float
    order_placed: bool
    order_filled: bool
    skipped_reason: str

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["timestamp"] = self.timestamp.isoformat()
        d["expansion_start"] = self.expansion_start.isoformat() if self.expansion_start is not None else ""
        d["expansion_end"] = self.expansion_end.isoformat() if self.expansion_end is not None else ""
        d["session_date"] = str(self.session_date)
        return d


# ---- main backtest --------------------------------------------------------

def run_backtest(cfg: CBRGoldScalpConfig,
                  m1: pd.DataFrame,
                  h1: pd.DataFrame,
                  dxy: pd.DataFrame | None = None,
                  *,
                  verbose: bool = False) -> dict:
    """Run a CBR-style scalp backtest end to end.

    Returns a dict with:
      trades:      list[TradeRow]
      setups:      list[SetupRow]
      validation:  data validation report
      runtime_s:   wall-clock seconds
    """
    t0 = _time.time()
    validation = _validate_data(cfg, m1, h1, dxy)

    # Annotate session columns
    window = SessionWindow.from_config(cfg.session)
    df = annotate_session_columns(m1, window)
    df = asia_session_high_low(df)
    df = attach_atr(df, cfg.expansion.atr_length, col="atr")

    # Optional date window
    if cfg.start_date:
        df = df[df["time"] >= pd.Timestamp(cfg.start_date, tz="UTC")]
    if cfg.end_date:
        df = df[df["time"] <  pd.Timestamp(cfg.end_date,   tz="UTC")]
    df = df.reset_index(drop=True)

    # Precompute bias frames
    h1_bias = precompute_h1_bias(h1, cfg.htf_bias)
    dxy_bias = precompute_dxy_bias(dxy, cfg.dxy)

    # Pivot pass: find ALL confirmed pivots up to the dataset end. The
    # engine then filters by `confirmed_at_idx <= current_idx` per bar.
    pivots = find_confirmed_pivots(
        df, pivot_left=cfg.structure.pivot_left,
        pivot_right=cfg.structure.pivot_right)

    trades: list[TradeRow] = []
    setups: list[SetupRow] = []
    state: SessionState | None = None
    equity = cfg.risk.initial_equity
    trade_id = 0

    n = len(df)
    # keep `time` as a Series so tz is preserved -- numpy conversion
    # via .values would strip the tz and break tz-aware comparisons in
    # resolve_h1_bias_at.
    times_series = df["time"]
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    sess_dates = df["session_date"].values
    in_asia = df["in_asia"].values
    in_exec = df["in_execution"].values
    asia_highs = df["asia_high_so_far"].values
    asia_lows = df["asia_low_so_far"].values
    atr = df["atr"].values

    for i in range(n):
        ts = times_series.iloc[i]    # tz-aware Timestamp
        sess_date = sess_dates[i]

        # ---- session boundary ----
        if state is None or state.session_date != sess_date:
            # close any open trade at session close = exit at open price next session start
            if state is not None and state.in_trade and state.plan is not None:
                trades.append(_close_trade(
                    cfg=cfg, state=state, exit_idx=i, exit_price=opens[i],
                    exit_reason="session_close",
                    df=df, trade_id=trade_id))
                trade_id += 1
            state = SessionState(session_date=sess_date)

        # ---- 1H bias resolution (no-lookahead) ----
        bias = resolve_h1_bias_at(ts, h1_bias)

        # ---- if in a trade, walk stop/target on this bar ----
        if state.in_trade and state.plan is not None:
            exit_event = _check_trade_exit(
                cfg=cfg, state=state, i=i,
                high=highs[i], low=lows[i], open_=opens[i],
                close=closes[i])
            if exit_event is not None:
                trades.append(_close_trade(
                    cfg=cfg, state=state, exit_idx=i,
                    exit_price=exit_event["price"],
                    exit_reason=exit_event["reason"],
                    df=df, trade_id=trade_id))
                trade_id += 1
                # IMPORTANT: clear in-trade state so subsequent bars
                # don't re-fire stop/target on the closed plan.
                state.in_trade = False
                state.plan = None
                state.trade_entry_idx = None
                state.trade_entry_price = None
                # done for the session unless one_trade_per_session is False
                if cfg.entry.one_trade_per_session:
                    state.state = "DONE"
                else:
                    # allow a fresh setup -- reset expansion / trigger / msb
                    state.expansion = None
                    state.trigger_idx = None
                    state.trigger_kind = None
                    state.trigger_direction = 0
                    state.sweep = None
                    state.rebalance = None
                    state.msb = None
                    state.state = "IDLE"

        # ---- expiring or filling a placed limit ----
        if state.state == "ENTRY_PLACED" and state.plan is not None:
            plan = state.plan
            # cancel on expiry
            if i > plan.expiry_bar_idx:
                state.plan = None
                state.state = "DONE" if cfg.entry.one_trade_per_session else "IDLE"
            else:
                fill = _check_limit_fill(plan, highs[i], lows[i])
                if fill:
                    state.in_trade = True
                    state.trade_entry_idx = i
                    state.trade_entry_price = plan.entry_price
                    state.state = "IN_TRADE"
                    if state.trades_taken_today is None:
                        state.trades_taken_today = 0
                    state.trades_taken_today += 1

        # ---- skip new setups outside the asia session entirely ----
        if not in_asia[i]:
            continue

        # ---- new-setup gate: warmup + execution window + per-day caps ----
        warmup_ok = i >= max(cfg.expansion.expansion_lookback_bars,
                              cfg.expansion.atr_length) + 5
        if not warmup_ok:
            continue
        # only ALLOW new triggers during exec window (unless config opens it)
        if not in_exec[i] and not cfg.session.allow_setup_outside_execution:
            continue
        if state.state in ("DONE", "IN_TRADE", "ENTRY_PLACED"):
            # while in trade or already entered, no new setups this session
            continue
        if state.trades_taken_today >= cfg.entry.max_trades_per_day:
            state.state = "DONE"
            continue

        # ---- 1) expansion ----
        if state.state == "IDLE" or state.expansion is None:
            exp = evaluate_expansion(df, end_idx=i - 1, cfg=cfg.expansion)
            if exp.detected:
                # quality threshold (optional)
                qt = cfg.expansion.quality_score_threshold
                if qt is not None and exp.quality_score < qt:
                    pass
                else:
                    state.expansion = exp
                    state.state = "EXPANSION_PENDING"

        # ---- 2) sweep / rebalance trigger ----
        if state.state == "EXPANSION_PENDING" and state.expansion is not None:
            exp = state.expansion
            sweep = SweepResult(detected=False)
            rebal = RebalanceResult(detected=False)
            mode = cfg.trigger.trigger_mode

            if mode in ("SWEEP_ONLY", "SWEEP_OR_REBALANCE", "SWEEP_AND_REBALANCE"):
                prev_h1_high, prev_h1_low = _resolve_prev_h1_levels(ts, h1)
                sweep = evaluate_sweep(df, i, prev_h1_high=prev_h1_high,
                                        prev_h1_low=prev_h1_low,
                                        lookback=30)
            if mode in ("REBALANCE_ONLY", "SWEEP_OR_REBALANCE", "SWEEP_AND_REBALANCE"):
                rebal = evaluate_rebalance(df, i, exp, cfg.trigger,
                                            atr_at_end=atr[i],
                                            tick_size=cfg.risk.tick_size)
            triggered = False
            tdir = 0
            tkind = ""
            if mode == "SWEEP_ONLY" and sweep.detected:
                triggered = True; tdir = sweep.direction; tkind = "sweep"
            elif mode == "REBALANCE_ONLY" and rebal.detected:
                triggered = True; tdir = rebal.direction; tkind = "rebalance"
            elif mode == "SWEEP_OR_REBALANCE":
                if sweep.detected:
                    triggered = True; tdir = sweep.direction; tkind = "sweep"
                elif rebal.detected:
                    triggered = True; tdir = rebal.direction; tkind = "rebalance"
            elif mode == "SWEEP_AND_REBALANCE":
                if sweep.detected and rebal.detected and sweep.direction == rebal.direction:
                    triggered = True; tdir = sweep.direction; tkind = "sweep+rebalance"
            if triggered:
                state.trigger_idx = i
                state.trigger_kind = tkind
                state.trigger_direction = tdir
                state.sweep = sweep
                state.rebalance = rebal
                state.state = "TRIGGER_PENDING"

        # ---- 3) MSB ----
        if state.state == "TRIGGER_PENDING":
            msb = evaluate_msb(
                df, end_idx=i, pivots=pivots, cfg=cfg.structure,
                trigger_idx=state.trigger_idx, trigger_direction=state.trigger_direction,
                max_bars_between=cfg.trigger.max_bars_between_trigger_and_msb,
            )
            # expire if too many bars passed
            if i > state.trigger_idx + cfg.trigger.max_bars_between_trigger_and_msb:
                _log_skip(setups, cfg, ts, sess_date, state, bias,
                           cfg.dxy.dxy_mode, "msb_timeout")
                state.state = "DONE" if cfg.entry.one_trade_per_session else "IDLE"
                state.expansion = None
                continue
            if msb.detected:
                # bias / DXY check
                if cfg.htf_bias.bias_mode != "OFF" and bias.direction != 0 \
                        and bias.direction != msb.direction:
                    _log_skip(setups, cfg, ts, sess_date, state, bias,
                               "no_dxy_check", "bias_against_msb_direction")
                    state.state = "DONE" if cfg.entry.one_trade_per_session else "IDLE"
                    state.expansion = None
                    continue
                dxy_ok, dxy_reason = dxy_inverse_check(
                    msb.direction, ts, dxy_bias, cfg.dxy)
                if not dxy_ok:
                    _log_skip(setups, cfg, ts, sess_date, state, bias,
                               dxy_reason, "dxy_against_msb_direction")
                    state.state = "DONE" if cfg.entry.one_trade_per_session else "IDLE"
                    state.expansion = None
                    continue
                if not in_exec[i] and not cfg.session.allow_entry_outside_execution:
                    _log_skip(setups, cfg, ts, sess_date, state, bias,
                               dxy_reason, "msb_outside_execution_window")
                    state.state = "DONE" if cfg.entry.one_trade_per_session else "IDLE"
                    state.expansion = None
                    continue

                # build entry plan
                prev_h1_open, prev_h1_close = _resolve_prev_h1_oc(ts, h1)
                plan = compute_entry_plan(
                    cfg=cfg, msb=msb, expansion=state.expansion,
                    prev_h1_open=prev_h1_open, prev_h1_close=prev_h1_close,
                    asia_high=asia_highs[i] if math.isfinite(asia_highs[i]) else float("nan"),
                    asia_low=asia_lows[i] if math.isfinite(asia_lows[i]) else float("nan"),
                    atr_at_entry=atr[i],
                    market_close_price=closes[i],
                    msb_idx=i,
                    equity=equity,
                )
                if plan is None:
                    _log_skip(setups, cfg, ts, sess_date, state, bias,
                               dxy_reason, "entry_plan_invalid")
                    state.state = "DONE" if cfg.entry.one_trade_per_session else "IDLE"
                    state.expansion = None
                    continue
                state.msb = msb
                state.plan = plan
                state.setup_count += 1
                if plan.fill_kind == "market":
                    fill_price = closes[i] if cfg.engine.market_entry_fill_convention == "close" \
                                              else (opens[i + 1] if i + 1 < n else closes[i])
                    state.plan.entry_price = fill_price
                    state.in_trade = True
                    state.trade_entry_idx = i
                    state.trade_entry_price = fill_price
                    state.trades_taken_today += 1
                    state.state = "IN_TRADE"
                else:
                    state.state = "ENTRY_PLACED"
                _log_setup(setups, cfg, ts, sess_date, state, bias, dxy_reason,
                            order_placed=True, order_filled=(plan.fill_kind == "market"),
                            skipped_reason="")

    # close any remaining open trade at end of data
    if state is not None and state.in_trade and state.plan is not None:
        last_close = float(closes[-1]) if n else float("nan")
        trades.append(_close_trade(
            cfg=cfg, state=state, exit_idx=n - 1,
            exit_price=last_close, exit_reason="dataset_end",
            df=df, trade_id=trade_id))

    return {
        "trades": trades,
        "setups": setups,
        "validation": validation,
        "runtime_s": round(_time.time() - t0, 1),
        "config_used": cfg.to_json(),
        "n_m1_bars": n,
        "n_h1_bars": len(h1),
    }


# ---- helpers --------------------------------------------------------------

def _check_limit_fill(plan: EntryPlan, high: float, low: float) -> bool:
    if plan.direction > 0:
        return low <= plan.entry_price
    return high >= plan.entry_price


def _check_trade_exit(*, cfg: CBRGoldScalpConfig, state: SessionState,
                       i: int, high: float, low: float,
                       open_: float, close: float) -> dict | None:
    plan = state.plan
    direction = plan.direction
    stop = plan.stop_price
    target = plan.target_price

    if direction > 0:
        hit_stop = low <= stop
        hit_target = high >= target
    else:
        hit_stop = high >= stop
        hit_target = low <= target

    if hit_stop and hit_target:
        # ambiguous: same-bar both hit. Default conservative = stop wins.
        if cfg.engine.ambiguous_bar_resolution == "OPTIMISTIC":
            return {"reason": "target_ambiguous_optimistic", "price": target}
        if cfg.engine.ambiguous_bar_resolution == "SKIP_TRADE":
            return {"reason": "ambiguous_skipped", "price": plan.entry_price}
        return {"reason": "stop_ambiguous_conservative", "price": stop}
    if hit_stop:
        return {"reason": "stop", "price": stop}
    if hit_target:
        return {"reason": "target", "price": target}
    return None


def _close_trade(*, cfg: CBRGoldScalpConfig, state: SessionState,
                  exit_idx: int, exit_price: float, exit_reason: str,
                  df: pd.DataFrame, trade_id: int) -> TradeRow:
    plan = state.plan
    entry_idx = state.trade_entry_idx
    entry_time = pd.Timestamp(df["time"].iloc[entry_idx])
    exit_time = pd.Timestamp(df["time"].iloc[exit_idx])
    direction = plan.direction

    pnl_per_unit = (exit_price - plan.entry_price) * direction
    pnl = pnl_per_unit * plan.quantity * (cfg.risk.tick_value / max(cfg.risk.tick_size, 1e-9))
    r = (pnl_per_unit / plan.risk_per_unit) if plan.risk_per_unit > 0 else 0.0

    sl = slice(entry_idx, exit_idx + 1)
    h_arr = df["high"].iloc[sl].values
    l_arr = df["low"].iloc[sl].values
    if direction > 0:
        mfe = float(h_arr.max() - plan.entry_price)
        mae = float(plan.entry_price - l_arr.min())
    else:
        mfe = float(plan.entry_price - l_arr.min())
        mae = float(h_arr.max() - plan.entry_price)

    setup_id = f"{state.session_date}_{state.setup_count}"
    return TradeRow(
        trade_id=trade_id, setup_id=setup_id,
        entry_time=entry_time, exit_time=exit_time,
        direction=direction,
        entry_price=plan.entry_price,
        stop_price=plan.stop_price,
        target_price=plan.target_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        r_result=round(r, 4),
        pnl=round(pnl, 2),
        duration_minutes=int((exit_time - entry_time).total_seconds() / 60),
        mae=round(mae, 5), mfe=round(mfe, 5),
        session_date=state.session_date,
        day_of_week=entry_time.day_name(),
        htf_bias=int(direction),  # placeholder; engine could store snapshot
        dxy_state="(see setup row)",
        expansion_quality=state.expansion.quality_score if state.expansion else 0.0,
        trigger_kind=state.trigger_kind or "",
        structure_mode=cfg.structure.structure_break_mode,
        retracement_level=(0.5 if cfg.entry.entry_mode == "LIMIT_50_RETRACE"
                           else 0.618 if cfg.entry.entry_mode == "LIMIT_618_RETRACE"
                           else cfg.entry.custom_retrace
                           if cfg.entry.entry_mode == "LIMIT_CUSTOM_RETRACE"
                           else 1.0),
    )


def _log_setup(setups: list, cfg, ts, sess_date, state, bias,
                dxy_reason: str, order_placed: bool, order_filled: bool,
                skipped_reason: str) -> None:
    setup_id = f"{sess_date}_{state.setup_count}"
    plan = state.plan
    setups.append(SetupRow(
        setup_id=setup_id,
        timestamp=ts,
        session_date=sess_date,
        symbol=cfg.symbol,
        direction=plan.direction if plan else 0,
        htf_bias=bias.direction,
        htf_bias_reason=bias.reason,
        dxy_state=dxy_reason,
        expansion_direction=state.expansion.direction if state.expansion else 0,
        expansion_quality=state.expansion.quality_score if state.expansion else 0.0,
        expansion_start=None, expansion_end=None,
        expansion_high=state.expansion.high if state.expansion else float("nan"),
        expansion_low=state.expansion.low if state.expansion else float("nan"),
        expansion_mid=state.expansion.midpoint if state.expansion else float("nan"),
        sweep_detected=bool(state.sweep and state.sweep.detected),
        swept_level=state.sweep.swept_level if state.sweep else float("nan"),
        rebalance_detected=bool(state.rebalance and state.rebalance.detected),
        rebalance_midpoint=state.rebalance.midpoint if state.rebalance else float("nan"),
        msb_detected=bool(state.msb and state.msb.detected),
        msb_break_level=state.msb.break_price if state.msb else float("nan"),
        entry_mode=cfg.entry.entry_mode,
        entry_price=plan.entry_price if plan else float("nan"),
        stop_price=plan.stop_price if plan else float("nan"),
        target_price=plan.target_price if plan else float("nan"),
        order_placed=order_placed,
        order_filled=order_filled,
        skipped_reason=skipped_reason,
    ))


def _log_skip(setups, cfg, ts, sess_date, state, bias,
               dxy_reason, skipped_reason):
    _log_setup(setups, cfg, ts, sess_date, state, bias, dxy_reason,
                order_placed=False, order_filled=False,
                skipped_reason=skipped_reason)


def _resolve_prev_h1_levels(ts: pd.Timestamp, h1: pd.DataFrame
                              ) -> tuple[float, float]:
    """Return (high, low) of the H1 bar that closed strictly before ts."""
    if h1.empty:
        return float("nan"), float("nan")
    end_times = h1["time"] + pd.Timedelta(hours=1)
    mask = end_times <= ts
    if not mask.any():
        return float("nan"), float("nan")
    last_idx = int(mask.values.nonzero()[0][-1])
    return float(h1["high"].iloc[last_idx]), float(h1["low"].iloc[last_idx])


def _resolve_prev_h1_oc(ts: pd.Timestamp, h1: pd.DataFrame
                          ) -> tuple[float, float]:
    if h1.empty:
        return float("nan"), float("nan")
    end_times = h1["time"] + pd.Timedelta(hours=1)
    mask = end_times <= ts
    if not mask.any():
        return float("nan"), float("nan")
    last_idx = int(mask.values.nonzero()[0][-1])
    return float(h1["open"].iloc[last_idx]), float(h1["close"].iloc[last_idx])


def _validate_data(cfg: CBRGoldScalpConfig,
                    m1: pd.DataFrame, h1: pd.DataFrame,
                    dxy: pd.DataFrame | None) -> dict:
    """Pre-flight validation. Returns a dict the runner writes to disk."""
    out: dict = {"issues": [], "warnings": [], "ok": True}
    if m1.empty or h1.empty:
        out["issues"].append("missing m1 or h1 data")
        out["ok"] = False
        return out
    # OHLC validity
    bad_h = ((m1["high"] < m1[["open", "close", "low"]].max(axis=1)) |
              (m1["low"] > m1[["open", "close", "high"]].min(axis=1))).sum()
    if bad_h:
        out["issues"].append(f"{int(bad_h)} m1 bars have invalid OHLC ordering")
    # negative prices
    neg = (m1[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if neg:
        out["issues"].append(f"{int(neg)} m1 bars have non-positive prices")
    # duplicate timestamps
    dups_m1 = int(m1["time"].duplicated().sum())
    dups_h1 = int(h1["time"].duplicated().sum())
    if dups_m1:
        out["issues"].append(f"{dups_m1} duplicate timestamps in m1")
    if dups_h1:
        out["issues"].append(f"{dups_h1} duplicate timestamps in h1")
    # tz consistency
    if m1["time"].iloc[0].tz is None:
        out["issues"].append("m1 timestamps are not tz-aware")
    if h1["time"].iloc[0].tz is None:
        out["issues"].append("h1 timestamps are not tz-aware")
    # length sanity
    if len(m1) < cfg.expansion.expansion_lookback_bars + cfg.expansion.atr_length + 5:
        out["issues"].append("m1 dataset shorter than expansion + ATR warmup")
    # DXY availability
    out["dxy_available"] = (dxy is not None and not dxy.empty)
    if cfg.dxy.dxy_mode != "OFF" and not out["dxy_available"]:
        out["warnings"].append(
            f"dxy_mode={cfg.dxy.dxy_mode} but no DXY data; "
            "DXY filter will be silently disabled")
    if out["issues"]:
        out["ok"] = False
    return out
