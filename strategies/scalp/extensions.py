"""Optional Batch-K2 filter extensions: ATR regime + news calendar.

These hook into the engine via `extensions.passes_extensions(...)`
which returns (ok, reason). If both filters are disabled the call
is a no-op and always passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.scalp.config import ATRRegimeConfig, NewsFilterConfig


# ---- ATR regime -----------------------------------------------------------

def attach_atr_percentile(df: pd.DataFrame, *,
                            atr_length: int = 14,
                            rolling_window: int = 720) -> pd.DataFrame:
    """Add `atr_pct` column = rolling-percentile rank of ATR over the
    last `rolling_window` bars. Read with `df["atr_pct"].iloc[i-1]`
    for no-lookahead use."""
    out = df.copy()
    if "atr" not in out.columns:
        # local TR + Wilder mean
        h = out["high"].values; l = out["low"].values
        c = out["close"].values
        pc = np.concatenate(([np.nan], c[:-1]))
        tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
        out["atr"] = pd.Series(tr).rolling(atr_length,
                                             min_periods=atr_length).mean().values
    out["atr_pct"] = (
        pd.Series(out["atr"].values)
        .rolling(rolling_window, min_periods=max(60, rolling_window // 4))
        .rank(pct=True)
        .values
    )
    return out


def passes_atr_regime(df: pd.DataFrame, idx: int,
                       cfg: ATRRegimeConfig) -> tuple[bool, str]:
    if not cfg.enabled:
        return True, "atr_regime_off"
    if "atr_pct" not in df.columns or idx <= 0:
        return True, "atr_regime_warmup"
    p = float(df["atr_pct"].iloc[idx - 1])    # shift-by-1
    if not math.isfinite(p):
        return True, "atr_regime_warmup"
    if p < cfg.min_percentile:
        return False, f"atr_regime_too_low_{p:.2f}"
    if p > cfg.max_percentile:
        return False, f"atr_regime_too_high_{p:.2f}"
    return True, f"atr_regime_ok_{p:.2f}"


# ---- News calendar --------------------------------------------------------

@dataclass
class NewsCalendar:
    events: pd.DataFrame                # columns: time (UTC), impact, symbol?
    cfg: NewsFilterConfig

    @classmethod
    def from_config(cls, cfg: NewsFilterConfig) -> "NewsCalendar":
        if not cfg.enabled:
            return cls(events=pd.DataFrame(columns=["time", "impact", "symbol"]),
                        cfg=cfg)
        if not cfg.csv_path:
            raise ValueError("news.enabled=True but csv_path not set")
        path = Path(cfg.csv_path)
        if not path.exists():
            # graceful degrade -- empty calendar, filter never blocks
            return cls(events=pd.DataFrame(columns=["time", "impact", "symbol"]),
                        cfg=cfg)
        df = pd.read_csv(path)
        if "time" not in df.columns:
            raise ValueError(f"news CSV missing 'time' column: {path}")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        if "impact" not in df.columns:
            df["impact"] = "high"
        if "symbol" not in df.columns:
            df["symbol"] = ""
        df = df.sort_values("time").reset_index(drop=True)
        return cls(events=df, cfg=cfg)

    def is_in_blackout(self, ts: pd.Timestamp,
                        symbol: str = "") -> tuple[bool, str]:
        if not self.cfg.enabled or self.events.empty:
            return False, "news_off" if not self.cfg.enabled else "news_calendar_empty"
        impacts = set(self.cfg.block_impacts)
        before = pd.Timedelta(minutes=self.cfg.window_minutes_before)
        after = pd.Timedelta(minutes=self.cfg.window_minutes_after)
        # narrow to events whose blackout window covers ts
        mask = (
            (self.events["time"] - before <= ts) &
            (self.events["time"] + after >= ts) &
            (self.events["impact"].astype(str).str.lower().isin([s.lower() for s in impacts]))
        )
        if symbol:
            mask &= ((self.events["symbol"] == "")
                     | (self.events["symbol"] == symbol)
                     | (self.events["symbol"] == "ALL"))
        if mask.any():
            ev = self.events.loc[mask].iloc[0]
            return True, f"news_blackout_{ev['impact']}_{ev['time']}"
        return False, "news_clear"


# ---- combined check used by the engine -------------------------------------

def passes_extensions(df: pd.DataFrame, idx: int, ts: pd.Timestamp,
                       atr_cfg: ATRRegimeConfig,
                       news_cal: NewsCalendar | None,
                       symbol: str = "") -> tuple[bool, str]:
    ok, reason = passes_atr_regime(df, idx, atr_cfg)
    if not ok:
        return False, reason
    if news_cal is not None and news_cal.cfg.enabled:
        in_blackout, news_reason = news_cal.is_in_blackout(ts, symbol)
        if in_blackout:
            return False, news_reason
    return True, "ext_ok"
