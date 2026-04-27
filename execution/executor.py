"""Realistic executor with slippage, spread widening, and missed fills.

Executes one strategy spec across a (h4, m15) pair and returns a list
of Trade objects. The executor:

  1. Resolves the directional bias from h4 (signal + filters).
  2. For each non-zero signal H4 bar:
        a. Calls the entry model on the next H4's M15 sub-bars.
        b. Applies an ExecutionModel to the candidate fill (slippage,
           spread widening, missed-fill probability).
        c. Walks the remaining sub-bars to determine stop / target hit.
  3. Records MAE / MFE / time-to-TP / time-to-SL on the trade.

The executor does NOT decide entries; entry models do. The executor
does NOT decide stops/exits; the spec does. Keeping these separate is
what lets us A/B compare entry models across the same dataset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.constants import POINT_SIZE
from core.types import Trade
from entry_models import registry as entry_registry


# ---------- execution model ----------

@dataclass
class ExecutionModel:
    slippage_bps_mean: float = 0.5      # typical slippage on market fills
    slippage_bps_vol: float = 1.0       # σ of slippage; price-volatility scaled
    spread_mult: float = 1.0            # 1.0 = use broker spread as-is; 1.5 = stress
    miss_prob_market: float = 0.00      # market orders almost always fill
    miss_prob_limit: float = 0.10       # limits sometimes get skipped
    near_miss_ticks: int = 1            # how close to a level still counts as near-miss
    rng_seed: int = 12345

    def stress(self) -> "ExecutionModel":
        return ExecutionModel(
            slippage_bps_mean=self.slippage_bps_mean * 2.0,
            slippage_bps_vol=self.slippage_bps_vol * 2.0,
            spread_mult=self.spread_mult * 1.5,
            miss_prob_market=self.miss_prob_market,
            miss_prob_limit=min(0.5, self.miss_prob_limit + 0.05),
            near_miss_ticks=self.near_miss_ticks,
            rng_seed=self.rng_seed,
        )


# ---------- signal + filters (v2) ----------

def _atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    out = pd.Series(tr).rolling(n, min_periods=n).mean().values
    return out


def _signal_series(h4: pd.DataFrame, signal_spec: dict) -> np.ndarray:
    if signal_spec["type"] != "prev_color":
        raise ValueError(f"unknown signal: {signal_spec['type']}")
    color = np.sign(h4["close"].values - h4["open"].values).astype(int)
    sig = np.empty_like(color)
    sig[0] = 0
    sig[1:] = color[:-1]
    return sig


def _apply_filters(h4: pd.DataFrame, sig: np.ndarray, filters: list[dict]) -> np.ndarray:
    n = len(h4)
    if n == 0:
        return sig
    o, c = h4["open"].values, h4["close"].values
    hi, lo = h4["high"].values, h4["low"].values
    hours = h4["time"].dt.hour.values
    mask = np.ones(n, dtype=bool)
    for f in filters:
        t = f["type"]
        if t == "body_atr":
            an = int(f.get("atr_n", 14))
            mn = float(f.get("min", 0.0))
            body_prev = np.empty(n); body_prev[0] = np.nan
            body_prev[1:] = np.abs(c[:-1] - o[:-1])
            atr_prev = np.empty(n); atr_prev[0] = np.nan
            atr_arr = _atr(h4, an); atr_prev[1:] = atr_arr[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = body_prev / atr_prev
            mask &= np.isfinite(ratio) & (ratio >= mn)
        elif t == "session":
            sess = set(int(h) for h in f["hours_utc"])
            mask &= np.isin(hours, np.array(list(sess), dtype=int))
        elif t == "regime":
            ma_n = int(f.get("ma_n", 50))
            side = f.get("side", "with")
            ma = pd.Series(c).rolling(ma_n).mean().shift(1).values
            prev_close = np.empty(n); prev_close[0] = np.nan
            prev_close[1:] = c[:-1]
            trend = np.sign(prev_close - ma)
            valid = np.isfinite(ma) & np.isfinite(prev_close)
            ok = (trend == sig) if side == "with" else (trend == -sig)
            mask &= valid & ok
        elif t == "min_streak":
            k = int(f.get("k", 2))
            color = np.sign(c - o).astype(int)
            ok = np.ones(n, dtype=bool)
            for i in range(1, k + 1):
                shifted = np.empty(n); shifted[:i] = 0; shifted[i:] = color[:-i]
                ok &= (shifted == sig)
            ok[:k] = False
            mask &= ok
        elif t == "atr_percentile":
            ap_n = int(f.get("window", 100))
            lo_p, hi_p = float(f.get("lo", 0.0)), float(f.get("hi", 1.0))
            atr_arr = _atr(h4, int(f.get("atr_n", 14)))
            ranks = pd.Series(atr_arr).rolling(ap_n).rank(pct=True).values
            mask &= np.isfinite(ranks) & (ranks >= lo_p) & (ranks <= hi_p)
        elif t == "vwap_dist":
            window = int(f.get("window", 24))
            max_z = float(f.get("max_z", 2.0))
            tp = (h4["high"].values + h4["low"].values + c) / 3.0
            v = h4["volume"].values
            num = pd.Series(tp * v).rolling(window).sum().values
            den = pd.Series(v).rolling(window).sum().values
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap = num / den
            std = pd.Series((tp - vwap) ** 2).rolling(window).mean().pow(0.5).values
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (c - vwap) / std
            mask &= np.isfinite(z) & (np.abs(z) <= max_z)
        else:
            raise ValueError(f"unknown filter: {t}")
    return np.where(mask, sig, 0)


# ---------- stops + exits ----------

def _stop_price(spec_stop: dict, h4: pd.DataFrame, i: int, sub: pd.DataFrame,
                entry_idx: int, entry_price: float, direction: int) -> float | None:
    s = spec_stop["type"]
    if s == "none":
        return None
    if s == "prev_h4_open":
        if i == 0:
            return None
        return float(h4["open"].iloc[i - 1])
    if s == "prev_h4_extreme":
        if i == 0:
            return None
        return float(h4["low"].iloc[i - 1] if direction > 0 else h4["high"].iloc[i - 1])
    if s == "h4_atr":
        an = int(spec_stop.get("atr_n", 14))
        mult = float(spec_stop.get("mult", 1.0))
        a = _atr(h4, an)
        a_prev = a[i - 1] if i > 0 else np.nan
        if not np.isfinite(a_prev):
            return None
        return entry_price - direction * mult * float(a_prev)
    if s == "m15_atr":
        an = int(spec_stop.get("atr_n", 14))
        mult = float(spec_stop.get("mult", 1.0))
        a = _atr(sub, an)
        if entry_idx >= len(a) or not np.isfinite(a[entry_idx]):
            return None
        return entry_price - direction * mult * float(a[entry_idx])
    raise ValueError(f"unknown stop: {s}")


def _exit_logic(spec_exit: dict, h4_row: pd.Series, prev_h4: pd.Series,
                direction: int) -> float | None:
    s = spec_exit["type"]
    if s == "h4_close":
        return float(h4_row["close"])
    if s == "prev_h4_extreme_tp":
        return float(prev_h4["high"] if direction > 0 else prev_h4["low"]) if prev_h4 is not None else None
    raise ValueError(f"unknown exit: {s}")


# ---------- main run ----------

def run(spec, h4: pd.DataFrame, m15: pd.DataFrame,
        execution: ExecutionModel | None = None) -> list[Trade]:
    """Execute a spec; return a list of Trade objects."""
    if execution is None:
        execution = ExecutionModel()
    rng = np.random.default_rng(execution.rng_seed)

    sig = _signal_series(h4, spec.signal)
    sig = _apply_filters(h4, sig, spec.filters)
    by_bucket: dict[pd.Timestamp, pd.DataFrame] = {
        k: g.sort_values("time").reset_index(drop=True)
        for k, g in m15.groupby(m15["time"].dt.floor("4h"), sort=False)
    }

    trades: list[Trade] = []
    h4_times = h4["time"]
    for i in range(len(h4)):
        s = int(sig[i])
        if s == 0:
            continue
        bucket = h4_times.iloc[i]
        sub = by_bucket.get(bucket)
        if sub is None or len(sub) == 0:
            continue
        prev_h4 = h4.iloc[i - 1] if i > 0 else None

        fill = entry_registry.fit(
            spec.entry["type"],
            h4.iloc[i], prev_h4, sub, s,
            spec.entry,
        )
        if fill is None:
            continue

        # --- missed fill probability for limit orders ---
        miss = execution.miss_prob_limit if fill.kind == "limit" else execution.miss_prob_market
        if miss > 0 and rng.random() < miss:
            continue

        # --- slippage on market fills (signed against trade) ---
        slip_price = 0.0
        if fill.kind == "market":
            slip_bps = execution.slippage_bps_mean + execution.slippage_bps_vol * abs(rng.normal())
            slip_price = fill.price * (slip_bps / 10_000.0)
        entry_price_filled = fill.price + s * slip_price  # adverse direction

        # --- spread paid on each leg ---
        sp_pts = float(sub["spread"].iloc[fill.sub_idx]) * execution.spread_mult
        spread_one_leg = sp_pts * POINT_SIZE
        # market orders cross the spread on entry; limits typically don't
        if fill.kind == "market":
            entry_price_filled += s * spread_one_leg

        # --- compute stop ---
        stop = _stop_price(spec.stop, h4, i, sub, fill.sub_idx, entry_price_filled, s)

        # --- compute target ---
        target = _exit_logic(spec.exit, h4.iloc[i], prev_h4, s)
        # target = float here for h4_close; for prev_h4_extreme_tp it's a level we exit at if touched

        # --- walk forward through sub bars to determine exit ---
        future = sub.iloc[fill.sub_idx:].reset_index(drop=True)
        exit_time = future["time"].iloc[-1]
        exit_price = float(h4.iloc[i]["close"])  # default exit at H4 close
        exit_reason = "h4_close"
        time_to_tp = None
        time_to_sl = None
        near_miss_tp = False
        mae = 0.0
        mfe = 0.0

        # Walk M15 bars to find first stop/target hit
        for j in range(len(future)):
            hi, lo = float(future["high"].iloc[j]), float(future["low"].iloc[j])
            # excursion tracking
            if s > 0:
                excursion_adv = hi - entry_price_filled
                excursion_adv = max(excursion_adv, 0.0)
                excursion_dis = entry_price_filled - lo
                excursion_dis = max(excursion_dis, 0.0)
            else:
                excursion_adv = entry_price_filled - lo
                excursion_adv = max(excursion_adv, 0.0)
                excursion_dis = hi - entry_price_filled
                excursion_dis = max(excursion_dis, 0.0)
            mfe = max(mfe, excursion_adv)
            mae = max(mae, excursion_dis)

            # stop check
            if stop is not None:
                if s > 0 and lo <= stop:
                    exit_time = future["time"].iloc[j]
                    exit_price = stop
                    exit_reason = "stop"
                    time_to_sl = (exit_time - fill_to_ts(future, 0, sub, fill)).total_seconds() / 60
                    break
                if s < 0 and hi >= stop:
                    exit_time = future["time"].iloc[j]
                    exit_price = stop
                    exit_reason = "stop"
                    time_to_sl = (exit_time - fill_to_ts(future, 0, sub, fill)).total_seconds() / 60
                    break

            # target check (only meaningful for prev_h4_extreme_tp)
            if spec.exit["type"] == "prev_h4_extreme_tp" and target is not None:
                if s > 0 and hi >= target:
                    exit_time = future["time"].iloc[j]
                    exit_price = target
                    exit_reason = "tp"
                    time_to_tp = (exit_time - fill_to_ts(future, 0, sub, fill)).total_seconds() / 60
                    break
                if s < 0 and lo <= target:
                    exit_time = future["time"].iloc[j]
                    exit_price = target
                    exit_reason = "tp"
                    time_to_tp = (exit_time - fill_to_ts(future, 0, sub, fill)).total_seconds() / 60
                    break

                # near-miss tracking: came within near_miss_ticks * point_size
                tol = execution.near_miss_ticks * POINT_SIZE
                if s > 0 and abs(target - hi) <= tol:
                    near_miss_tp = True
                if s < 0 and abs(target - lo) <= tol:
                    near_miss_tp = True

        # --- spread paid on exit (always, since we're closing position) ---
        exit_price_filled = exit_price - s * spread_one_leg

        gross = s * (exit_price_filled - entry_price_filled)
        cost = 2 * spread_one_leg + slip_price  # bookkeeping only
        pnl = gross  # spreads & slip already in entry/exit prices
        ret = pnl / entry_price_filled

        t = Trade(
            entry_time=fill_to_ts(future, 0, sub, fill),
            exit_time=exit_time,
            direction=s,
            entry=float(entry_price_filled),
            exit=float(exit_price_filled),
            cost=float(cost),
            pnl=float(pnl),
            ret=float(ret),
            mae=float(mae),
            mfe=float(mfe),
            time_to_tp_min=float(time_to_tp) if time_to_tp is not None else None,
            time_to_sl_min=float(time_to_sl) if time_to_sl is not None else None,
            near_miss_tp=bool(near_miss_tp),
            fill_kind=fill.kind,
            slippage=float(slip_price),
            spread_paid=float(spread_one_leg),
            h4_bucket=bucket,
            extras={"exit_reason": exit_reason, "entry_notes": fill.notes,
                    "stop": stop, "target": target},
        )
        trades.append(t)
    return trades


def fill_to_ts(future: pd.DataFrame, j: int, sub: pd.DataFrame, fill) -> pd.Timestamp:
    """Helper: timestamp of the entry sub-bar."""
    return sub["time"].iloc[fill.sub_idx]
