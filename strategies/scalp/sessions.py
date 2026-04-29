"""Session window logic for the CBR scalp engine.

Dukascopy timestamps are UTC. The strategy expresses session windows
in a configured local timezone (default Australia/Melbourne). We
convert UTC -> local at query time so daylight savings is handled
implicitly by `zoneinfo`.

Two windows per session day:

  * `asia_session`      [start, end)   -- the tracked session (high/low,
                                          extremes used by some target modes)
  * `execution_window`  [start, end)   -- when new setups can fire

The two windows must obey: asia_start <= exec_start <= exec_end <= asia_end.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from strategies.scalp.config import SessionConfig


@dataclass
class SessionWindow:
    timezone: ZoneInfo
    asia_start: _dt.time
    asia_end: _dt.time
    exec_start: _dt.time
    exec_end: _dt.time

    @classmethod
    def from_config(cls, cfg: SessionConfig) -> "SessionWindow":
        tz = ZoneInfo(cfg.timezone)
        a_start = _dt.time.fromisoformat(cfg.asia_session_start)
        a_end = _dt.time.fromisoformat(cfg.asia_session_end)
        e_start = _dt.time.fromisoformat(cfg.execution_window_start)
        e_end = _dt.time.fromisoformat(cfg.execution_window_end)
        if not (a_start <= e_start and e_end <= a_end):
            raise ValueError(
                f"execution window {e_start}-{e_end} must be inside "
                f"asia session {a_start}-{a_end}")
        return cls(tz, a_start, a_end, e_start, e_end)

    # ---- per-bar checks ----------------------------------------------

    def in_asia(self, ts_utc: pd.Timestamp) -> bool:
        local = ts_utc.tz_convert(self.timezone)
        return self.asia_start <= local.time() < self.asia_end

    def in_execution(self, ts_utc: pd.Timestamp) -> bool:
        local = ts_utc.tz_convert(self.timezone)
        return self.exec_start <= local.time() < self.exec_end

    def session_date(self, ts_utc: pd.Timestamp) -> _dt.date:
        """One unique date per session. The asia session never
        crosses midnight under default settings, so this is just
        the local-tz calendar date of the bar."""
        local = ts_utc.tz_convert(self.timezone)
        return local.date()


def annotate_session_columns(df: pd.DataFrame,
                              window: SessionWindow) -> pd.DataFrame:
    """Vectorised pre-compute of `in_asia`, `in_execution`,
    `session_date` columns. The engine reads these instead of calling
    `window.*` per-bar, which is 30-50x faster on 2M-bar frames."""
    out = df.copy()
    if "time" not in out.columns:
        raise ValueError("df missing required 'time' column")
    local = out["time"].dt.tz_convert(window.timezone)
    out["session_date"] = local.dt.date
    out["session_local_time"] = local.dt.time
    out["in_asia"] = (out["session_local_time"] >= window.asia_start) & \
                       (out["session_local_time"] < window.asia_end)
    out["in_execution"] = (out["session_local_time"] >= window.exec_start) & \
                           (out["session_local_time"] < window.exec_end)
    out = out.drop(columns=["session_local_time"])
    return out


def asia_session_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """For each bar in `df`, attach `asia_high_so_far` and
    `asia_low_so_far` -- the running session extremes including the
    current bar. Bars outside the asia session get NaN (so callers
    don't accidentally use last-session values across a midnight
    boundary)."""
    out = df.copy()
    out["asia_high_so_far"] = float("nan")
    out["asia_low_so_far"] = float("nan")
    if "in_asia" not in out.columns or "session_date" not in out.columns:
        raise ValueError("annotate_session_columns must be called first")

    in_asia = out["in_asia"].values
    sess_date = out["session_date"].values
    high = out["high"].values
    low = out["low"].values
    a_high = out["asia_high_so_far"].values.copy()
    a_low = out["asia_low_so_far"].values.copy()

    cur_high = float("nan")
    cur_low = float("nan")
    cur_date = None
    for i in range(len(out)):
        if in_asia[i]:
            if sess_date[i] != cur_date:
                cur_date = sess_date[i]
                cur_high = high[i]
                cur_low = low[i]
            else:
                cur_high = max(cur_high, high[i])
                cur_low = min(cur_low, low[i])
            a_high[i] = cur_high
            a_low[i] = cur_low
    out["asia_high_so_far"] = a_high
    out["asia_low_so_far"] = a_low
    return out
