"""Strategy DSL + runners for the 4h-continuation search.

A strategy is a JSON-serialisable dict (the "spec"). Two runners consume it:

  run_h4_sim(spec, h4)        -- whole-bar simulation on H4 only.
                                 Used by the walk-forward harness across
                                 the long history (M15 is unavailable
                                 pre-2026). Vectorised numpy.

  run_full_sim(spec, h4, m15) -- M15-aware simulation matching the
                                 spec.entry mode. Used for the holdout
                                 on the matched 2026 window with broker
                                 spread costs. Vectorised where possible
                                 with a small per-bar Python loop only
                                 for stop scans.

Both return a list[Trade] with identical fields, so metrics are computed
the same way regardless of which sim ran.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd


H4_BARS_PER_YEAR = 1560
H4_HOURS = 4
POINT_SIZE = 0.001  # XAUUSDc point size (broker spec)


# ---------- spec schema ----------

DEFAULT_SPEC: dict = {
    "id": "baseline",
    "signal": {"type": "prev_color"},
    "filters": [],
    # entry types: h4_open, m15_open, m15_confirm, m15_atr_stop, m15_retrace_50
    "entry": {"type": "h4_open"},
    # stop types: none, h4_atr, m15_atr, prev_h4_open, prev_h4_extreme
    "stop": {"type": "none"},
    # exit types: h4_close, prev_h4_extreme_tp (TP at prev H4 hi/lo, else H4 close)
    "exit": {"type": "h4_close"},
    # cost_model: 'spread' uses M15 broker spread; 'bps' uses cost_bps regardless.
    "cost_model": "spread",
    "cost_bps": 1.5,
}


def merge_spec(spec: dict) -> dict:
    out = json.loads(json.dumps(DEFAULT_SPEC))
    for k, v in spec.items():
        out[k] = v
    return out


# ---------- shared types + helpers ----------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry: float
    exit: float
    cost: float
    pnl: float
    ret: float


def atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    pc = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - pc),
        np.abs(low - pc),
    ])
    out = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= n:
        c = np.cumsum(np.where(np.isnan(tr), 0.0, tr))
        c[n:] = c[n:] - c[:-n]
        out[n - 1:] = c[n - 1:] / n
        # mask any window containing NaN
        nanmask = np.isnan(tr)
        bad = np.zeros_like(nanmask, dtype=bool)
        for i in range(n - 1, len(tr)):
            if nanmask[i - n + 1:i + 1].any():
                bad[i] = True
        out[bad] = np.nan
    return out


def trades_to_metrics(name: str, trades: list[Trade]) -> dict:
    if not trades:
        return {"strategy": name, "trades": 0, "wins": 0, "win_rate": 0.0,
                "avg_ret_bp": 0.0, "total_return": 0.0, "sharpe_ann": 0.0,
                "max_drawdown": 0.0, "avg_hold_min": 0.0}
    rets = np.array([t.ret for t in trades], dtype=float)
    eq = np.cumprod(1.0 + rets)
    wins = int((rets > 0).sum())
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (rets.mean() / sd) * math.sqrt(H4_BARS_PER_YEAR) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min()) if len(eq) else 0.0
    hold = float(np.mean([
        (t.exit_time - t.entry_time).total_seconds() / 60.0 for t in trades
    ]))
    return {
        "strategy": name,
        "trades": len(trades),
        "wins": wins,
        "win_rate": round(wins / len(rets), 4),
        "avg_ret_bp": round(float(rets.mean() * 10_000), 3),
        "total_return": round(float(eq[-1] - 1.0), 4),
        "sharpe_ann": round(float(sharpe), 3),
        "max_drawdown": round(dd, 4),
        "avg_hold_min": round(hold, 1),
    }


# ---------- signal + filters (vectorised) ----------

def compute_signal_np(h4: pd.DataFrame, spec: dict) -> np.ndarray:
    if spec["signal"]["type"] != "prev_color":
        raise ValueError(f"unknown signal: {spec['signal']['type']}")
    color = np.sign(h4["close"].values - h4["open"].values).astype(int)
    sig = np.empty_like(color)
    sig[0] = 0
    sig[1:] = color[:-1]
    return sig


def apply_filters_np(h4: pd.DataFrame, sig: np.ndarray, filters: Iterable[dict]) -> np.ndarray:
    n = len(h4)
    if n == 0:
        return sig
    o = h4["open"].values
    c = h4["close"].values
    hi = h4["high"].values
    lo = h4["low"].values
    hours = h4["time"].dt.hour.values
    mask = np.ones(n, dtype=bool)

    for f in filters:
        t = f["type"]
        if t == "body_atr":
            an = int(f.get("atr_n", 14))
            mn = float(f.get("min", 0.0))
            body_prev = np.empty(n)
            body_prev[0] = np.nan
            body_prev[1:] = np.abs(c[:-1] - o[:-1])
            a_prev = np.empty(n)
            a = atr_np(hi, lo, c, an)
            a_prev[0] = np.nan
            a_prev[1:] = a[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = body_prev / a_prev
            ok = np.isfinite(ratio) & (ratio >= mn)
            mask &= ok
        elif t == "session":
            sess = set(int(h) for h in f["hours_utc"])
            sess_arr = np.array(list(sess), dtype=int)
            mask &= np.isin(hours, sess_arr)
        elif t == "regime":
            ma_n = int(f.get("ma_n", 50))
            side = f.get("side", "with")
            ma = pd.Series(c).rolling(ma_n).mean().shift(1).values
            prev_close = np.empty(n)
            prev_close[0] = np.nan
            prev_close[1:] = c[:-1]
            trend = np.sign(prev_close - ma)
            valid = np.isfinite(ma) & np.isfinite(prev_close)
            if side == "with":
                ok = (trend == sig)
            else:
                ok = (trend == -sig)
            mask &= valid & ok
        elif t == "min_streak":
            k = int(f.get("k", 2))
            color = np.sign(c - o).astype(int)
            ok = np.ones(n, dtype=bool)
            for i in range(1, k + 1):
                shifted = np.empty(n)
                shifted[:i] = 0
                shifted[i:] = color[:-i]
                ok &= (shifted == sig)
            ok[:k] = False
            mask &= ok
        elif t == "candle_class":
            # classify the *previous* H4 candle as trend / rotation / exhaustion.
            cls = f.get("classes", ["trend"])
            body_min = float(f.get("body_min", 0.6))      # body/range threshold for "trend"
            close_pct = float(f.get("close_pct", 0.20))   # close in top/bottom this fraction of range
            wick_min = float(f.get("wick_min", 0.6))      # max wick/range threshold for "exhaustion"
            rng = (hi - lo)
            body = np.abs(c - o)
            with np.errstate(divide="ignore", invalid="ignore"):
                body_ratio = np.where(rng > 0, body / rng, 0.0)
                upper_wick = (hi - np.maximum(o, c)) / np.where(rng > 0, rng, 1.0)
                lower_wick = (np.minimum(o, c) - lo) / np.where(rng > 0, rng, 1.0)
                close_in_range = np.where(rng > 0, (c - lo) / rng, 0.5)
            is_trend_up = (body_ratio >= body_min) & (close_in_range >= 1.0 - close_pct)
            is_trend_dn = (body_ratio >= body_min) & (close_in_range <= close_pct)
            is_trend = is_trend_up | is_trend_dn
            is_rotation = (body_ratio < 0.4) & (np.abs(close_in_range - 0.5) < 0.20)
            is_exhaustion = (np.maximum(upper_wick, lower_wick) >= wick_min)
            tag = np.where(is_trend, "trend",
                  np.where(is_exhaustion, "exhaustion",
                  np.where(is_rotation, "rotation", "other")))
            tag_prev = np.empty(n, dtype=object); tag_prev[0] = "other"
            tag_prev[1:] = tag[:-1]
            mask &= np.isin(tag_prev, list(cls))
        else:
            raise ValueError(f"unknown filter: {t}")
    return np.where(mask, sig, 0)


# ---------- H4-only simulation (used by walk-forward) ----------

def run_h4_sim(spec: dict, h4: pd.DataFrame, return_diag: bool = False):
    """H4-only whole-bar simulation.

    Used for walk-forward where M15 is unavailable. The retracement and
    structural-stop entry types are M15-specific and are *skipped here*;
    that is the right behaviour because the walk-forward gates the
    *signal* edge, not the M15 execution refinement.

    If return_diag is True, returns (trades, diag) with skip counters.
    """
    spec = merge_spec(spec)
    h4 = h4.sort_values("time").reset_index(drop=True)
    n = len(h4)
    if n < 2:
        return ([], {"h4_bars": n, "signal_zero": n}) if return_diag else []

    o = h4["open"].values.astype(float)
    cl = h4["close"].values.astype(float)
    hi = h4["high"].values.astype(float)
    lo = h4["low"].values.astype(float)
    times = h4["time"].reset_index(drop=True)  # tz-aware

    sig = compute_signal_np(h4, spec)
    sig = apply_filters_np(h4, sig, spec["filters"])
    cost_bps = float(spec.get("cost_bps", 1.5))

    # default exit: H4 close
    exit_price = cl.copy()

    stop = spec["stop"]
    if stop["type"] == "h4_atr":
        an = int(stop.get("atr_n", 14))
        mult = float(stop.get("mult", 1.0))
        a = atr_np(hi, lo, cl, an)
        a_prev = np.empty(n)
        a_prev[0] = np.nan
        a_prev[1:] = a[:-1]
        stop_price = o - sig * mult * a_prev
        with np.errstate(invalid="ignore"):
            hit_long = (sig > 0) & np.isfinite(stop_price) & (lo <= stop_price)
            hit_short = (sig < 0) & np.isfinite(stop_price) & (hi >= stop_price)
        exit_price = np.where(hit_long | hit_short, stop_price, cl)

    take = sig != 0
    if not take.any():
        return []

    take_idx = np.where(take)[0]
    cost_price = o[take_idx] * (cost_bps / 10_000.0)
    gross = sig[take_idx] * (exit_price[take_idx] - o[take_idx])
    pnl = gross - cost_price
    ret = pnl / o[take_idx]

    out: list[Trade] = []
    for k, idx in enumerate(take_idx):
        ts = times.iloc[int(idx)]
        out.append(Trade(
            entry_time=ts,
            exit_time=ts,
            direction=int(sig[idx]),
            entry=float(o[idx]),
            exit=float(exit_price[idx]),
            cost=float(cost_price[k]),
            pnl=float(pnl[k]),
            ret=float(ret[k]),
        ))
    if return_diag:
        diag = {"h4_bars": n, "signal_zero": int(n - take.sum()),
                "trades": int(take.sum())}
        return out, diag
    return out


# ---------- M15-aware simulation (used by holdout) ----------

def _bucket(t: pd.Series) -> pd.Series:
    return t.dt.floor(f"{H4_HOURS}h")


def run_full_sim(spec: dict, h4: pd.DataFrame, m15: pd.DataFrame,
                 return_diag: bool = False):
    """Run the spec on matched M15 + H4 data.

    If return_diag is True, returns (trades, diag) where diag is a dict
    of skip-reason counters useful for the audit and for reproducible
    pipeline observability.
    """
    spec = merge_spec(spec)
    h4 = h4.sort_values("time").reset_index(drop=True)
    m15 = m15.sort_values("time").reset_index(drop=True).copy()

    sig = compute_signal_np(h4, spec)
    sig = apply_filters_np(h4, sig, spec["filters"])
    h4_open = h4["open"].values.astype(float)
    h4_high = h4["high"].values.astype(float)
    h4_low = h4["low"].values.astype(float)
    h4_close = h4["close"].values.astype(float)
    h4_time_series = h4["time"].reset_index(drop=True)

    m15["bucket"] = _bucket(m15["time"])
    atr_n = int(spec.get("stop", {}).get("atr_n", 14))
    m15["m15_atr"] = atr_np(
        m15["high"].values.astype(float),
        m15["low"].values.astype(float),
        m15["close"].values.astype(float),
        atr_n,
    )
    by_bucket: dict[pd.Timestamp, pd.DataFrame] = {
        k: g.reset_index(drop=True) for k, g in m15.groupby("bucket", sort=False)
    }

    entry_mode = spec["entry"]["type"]
    stop_mode = spec["stop"]["type"]
    stop_mult = float(spec["stop"].get("mult", 1.0))
    exit_mode = spec["exit"]["type"]
    cost_model = spec.get("cost_model", "spread")
    cost_bps = float(spec.get("cost_bps", 1.5))

    # Previous H4 levels (for retracement entries / structural stops / TPs).
    prev_open = np.empty(len(h4)); prev_open[0] = np.nan; prev_open[1:] = h4_open[:-1]
    prev_high = np.empty(len(h4)); prev_high[0] = np.nan; prev_high[1:] = h4_high[:-1]
    prev_low  = np.empty(len(h4)); prev_low[0]  = np.nan; prev_low[1:]  = h4_low[:-1]
    prev_mid = (prev_high + prev_low) / 2.0

    h4_atr_arr = atr_np(h4_high, h4_low, h4_close, int(spec["stop"].get("atr_n", 14)))
    h4_atr_prev = np.empty(len(h4)); h4_atr_prev[0] = np.nan
    h4_atr_prev[1:] = h4_atr_arr[:-1]

    diag = {
        "h4_bars": len(h4),
        "signal_zero": 0,
        "no_subbars": 0,
        "no_confirm": 0,
        "no_retrace": 0,
        "missing_prev_levels": 0,
    }

    out: list[Trade] = []
    for i in range(len(h4)):
        s = int(sig[i])
        if s == 0:
            diag["signal_zero"] += 1
            continue
        sub = by_bucket.get(h4_time_series.iloc[i])
        if sub is None or len(sub) == 0:
            diag["no_subbars"] += 1
            continue

        # ----- entry -----
        if entry_mode in ("h4_open", "m15_open", "m15_atr_stop"):
            entry_idx = 0
            entry_price = float(sub["open"].iloc[0])
            entry_time = sub["time"].iloc[0]
        elif entry_mode == "m15_confirm":
            sub_color = np.sign(sub["close"].values - sub["open"].values).astype(int)
            mhits = np.where(sub_color == s)[0]
            if len(mhits) == 0:
                diag["no_confirm"] += 1
                continue
            entry_idx = int(mhits[0])
            entry_price = float(sub["close"].iloc[entry_idx])
            entry_time = sub["time"].iloc[entry_idx]
        elif entry_mode == "m15_retrace_50":
            mid = float(prev_mid[i])
            if math.isnan(mid):
                diag["missing_prev_levels"] += 1
                continue
            sub_lo = sub["low"].values
            sub_hi = sub["high"].values
            if s > 0:
                hits = np.where(sub_lo <= mid)[0]
            else:
                hits = np.where(sub_hi >= mid)[0]
            if len(hits) == 0:
                diag["no_retrace"] += 1
                continue
            entry_idx = int(hits[0])
            # Fill at the midpoint (limit-style).
            entry_price = mid
            entry_time = sub["time"].iloc[entry_idx]
        else:
            raise ValueError(f"unknown entry mode: {entry_mode}")

        # ----- stop -----
        stop_price: float | None = None
        if stop_mode == "m15_atr" or entry_mode == "m15_atr_stop":
            a = float(sub["m15_atr"].iloc[entry_idx])
            if not math.isnan(a):
                mult = stop_mult if stop_mode == "m15_atr" else 1.0
                stop_price = entry_price - s * mult * a
        elif stop_mode == "h4_atr":
            a = h4_atr_prev[i]
            if not math.isnan(a):
                stop_price = entry_price - s * stop_mult * a
        elif stop_mode == "prev_h4_open":
            po = float(prev_open[i])
            if not math.isnan(po):
                stop_price = po
        elif stop_mode == "prev_h4_extreme":
            ext = float(prev_low[i]) if s > 0 else float(prev_high[i])
            if not math.isnan(ext):
                stop_price = ext

        # ----- target (exit) -----
        target_price: float | None = None
        if exit_mode == "prev_h4_extreme_tp":
            ext = float(prev_high[i]) if s > 0 else float(prev_low[i])
            if not math.isnan(ext):
                target_price = ext

        # ----- forward scan: stop / target / time exit -----
        future_hi = sub["high"].values[entry_idx:]
        future_lo = sub["low"].values[entry_idx:]
        future_t = sub["time"].iloc[entry_idx:].reset_index(drop=True)

        exit_time = sub["time"].iloc[-1]
        exit_price = float(h4_close[i])

        first_stop = None
        if stop_price is not None:
            if s > 0:
                idxs = np.where(future_lo <= stop_price)[0]
            else:
                idxs = np.where(future_hi >= stop_price)[0]
            if len(idxs) > 0:
                first_stop = int(idxs[0])

        first_tp = None
        if target_price is not None:
            if s > 0:
                idxs = np.where(future_hi >= target_price)[0]
            else:
                idxs = np.where(future_lo <= target_price)[0]
            if len(idxs) > 0:
                first_tp = int(idxs[0])

        if first_stop is not None and (first_tp is None or first_stop <= first_tp):
            exit_time = future_t.iloc[first_stop]
            exit_price = float(stop_price)
        elif first_tp is not None:
            exit_time = future_t.iloc[first_tp]
            exit_price = float(target_price)

        # ----- cost -----
        if cost_model == "bps":
            cost_price = entry_price * (cost_bps / 10_000.0)
        else:  # 'spread'
            sp = float(sub["spread"].iloc[entry_idx]) + float(sub["spread"].iloc[-1])
            cost_price = sp * POINT_SIZE

        gross = s * (exit_price - entry_price)
        pnl = gross - cost_price

        out.append(Trade(
            entry_time=entry_time, exit_time=exit_time, direction=s,
            entry=float(entry_price), exit=float(exit_price), cost=float(cost_price),
            pnl=float(pnl), ret=float(pnl / entry_price),
        ))

    if return_diag:
        return out, diag
    return out


# ---------- spec-id helper ----------

def spec_id(spec: dict) -> str:
    parts = []
    for f in spec.get("filters", []):
        if f["type"] == "body_atr":
            parts.append(f"body{f.get('min', 0)}")
        elif f["type"] == "session":
            parts.append("ses" + "-".join(str(h) for h in f["hours_utc"]))
        elif f["type"] == "regime":
            parts.append(f"reg{f.get('ma_n', 50)}{f.get('side', 'with')[:3]}")
        elif f["type"] == "min_streak":
            parts.append(f"streak{f.get('k', 2)}")
        elif f["type"] == "candle_class":
            parts.append("cls" + "-".join(f.get("classes", ["trend"])))
    parts.append(spec["entry"]["type"])
    s = spec["stop"]
    if s["type"] != "none":
        if "mult" in s:
            parts.append(f"{s['type']}x{s.get('mult', 1.0)}")
        else:
            parts.append(s["type"])
    e = spec["exit"]
    if e["type"] != "h4_close":
        parts.append(e["type"])
    return "_".join(parts) if parts else "raw"
