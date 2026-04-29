"""Per-H4-bar TPO level lookup + executor filter dispatcher.

The Batch G TPO families (`tpo_value_rejection`, `tpo_poc_reversion`)
emit candidates whose filters reference TPO tokens (`tpo_poc`,
`tpo_vah`, `tpo_val`, `tpo_value_acceptance`, `tpo_value_rejection`,
`tpo_poor_high`, `tpo_poor_low`, `tpo_excess`, `tpo_single_print`).
This module:

  1. Pre-computes the previous session's TPO profile for every H4
     bar and attaches `prev_session_tpo_{poc, vah, val, ib_high,
     ib_low, poor_high, poor_low, excess_high, excess_low}` columns
     to a copy of the H4 frame. Use `attach_prev_session_tpo`
     before passing the frame to the executor.

  2. Implements the executor's TPO filter dispatch via
     `apply_tpo_filter(token, h4, mask, sig, shift1, params)`. The
     executor's `_apply_filters` calls this when it sees any
     `tpo_*` token. The dispatcher reads `prev_session_tpo_*`
     columns; if they are missing it returns `mask & False` so the
     strategy produces no trades (we refuse to silently let trades
     through without the filter actually firing).

The TPO computation uses `analytics.tpo.compute_tpo_profile` with
defaults appropriate for XAUUSD (bin_size=0.50, bracket_minutes=30,
NY session 13:00-20:00 UTC). Callers can override.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.tpo import compute_tpo_profile, session_slices


PREV_SESSION_TPO_COLUMNS = (
    "prev_session_tpo_poc",
    "prev_session_tpo_vah",
    "prev_session_tpo_val",
    "prev_session_tpo_ib_high",
    "prev_session_tpo_ib_low",
    "prev_session_tpo_poor_high",   # bool
    "prev_session_tpo_poor_low",    # bool
    "prev_session_tpo_excess_high", # bool
    "prev_session_tpo_excess_low",  # bool
)


def attach_prev_session_tpo(h4: pd.DataFrame, m15: pd.DataFrame,
                             *,
                             session_start_hour_utc: int = 13,
                             session_length_hours: int = 7,
                             bin_size: float = 0.50,
                             bracket_minutes: int = 30,
                             ) -> pd.DataFrame:
    """Return a COPY of `h4` with `prev_session_tpo_*` columns added.

    For each H4 bar, the "previous session" is the most recent
    completed session (window of `session_length_hours` starting at
    `session_start_hour_utc` UTC) whose end time is on or before
    the H4 bar's start time. The TPO profile is computed from M15
    candles inside that prior session.

    Bars that have no prior session yet (early in the dataset) get
    NaN for numeric columns and False for bool columns.
    """
    out = h4.sort_values("time").reset_index(drop=True).copy()
    if m15.empty:
        for col in PREV_SESSION_TPO_COLUMNS:
            if col.endswith(("poor_high", "poor_low",
                              "excess_high", "excess_low")):
                out[col] = False
            else:
                out[col] = np.nan
        return out

    m15 = m15.sort_values("time").reset_index(drop=True)
    # Build a list of (session_end_ts, profile) tuples ordered by
    # session_end. For each H4 bar we then bisect.
    sessions: list[tuple[pd.Timestamp, dict]] = []
    end_offset = pd.Timedelta(hours=session_length_hours)
    for sess_df in session_slices(
            m15,
            session_start_hour_utc=session_start_hour_utc,
            session_length_hours=session_length_hours):
        if sess_df.empty:
            continue
        prof = compute_tpo_profile(
            sess_df, bin_size=bin_size,
            bracket_minutes=bracket_minutes)
        # session_end = first bar's session-day midnight + start_hour + length
        first = sess_df["time"].iloc[0]
        sess_start = pd.Timestamp(first.year, first.month, first.day,
                                    session_start_hour_utc,
                                    tz=first.tzinfo)
        sess_end = sess_start + end_offset
        sessions.append((sess_end, prof))
    sessions.sort(key=lambda t: t[0])
    sess_ends = np.array([s[0].value for s in sessions], dtype=np.int64)

    # For each H4 bar's start time, find the latest session_end <= that.
    bar_starts = out["time"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(sess_ends, bar_starts, side="right") - 1

    cols = {c: [] for c in PREV_SESSION_TPO_COLUMNS}
    for j in idx:
        if j < 0:
            cols["prev_session_tpo_poc"].append(np.nan)
            cols["prev_session_tpo_vah"].append(np.nan)
            cols["prev_session_tpo_val"].append(np.nan)
            cols["prev_session_tpo_ib_high"].append(np.nan)
            cols["prev_session_tpo_ib_low"].append(np.nan)
            cols["prev_session_tpo_poor_high"].append(False)
            cols["prev_session_tpo_poor_low"].append(False)
            cols["prev_session_tpo_excess_high"].append(False)
            cols["prev_session_tpo_excess_low"].append(False)
            continue
        prof = sessions[j][1]
        cols["prev_session_tpo_poc"].append(prof.poc)
        cols["prev_session_tpo_vah"].append(prof.vah)
        cols["prev_session_tpo_val"].append(prof.val)
        cols["prev_session_tpo_ib_high"].append(
            prof.initial_balance_high if prof.initial_balance_high is not None else np.nan)
        cols["prev_session_tpo_ib_low"].append(
            prof.initial_balance_low if prof.initial_balance_low is not None else np.nan)
        cols["prev_session_tpo_poor_high"].append(prof.poor_high)
        cols["prev_session_tpo_poor_low"].append(prof.poor_low)
        cols["prev_session_tpo_excess_high"].append(prof.excess_high)
        cols["prev_session_tpo_excess_low"].append(prof.excess_low)
    for c, vals in cols.items():
        out[c] = vals
    return out


# ---------------- executor filter dispatcher ----------------

def _have_columns(h4: pd.DataFrame, *cols: str) -> bool:
    return all(c in h4.columns for c in cols)


def apply_tpo_filter(token: str, h4: pd.DataFrame, mask: np.ndarray,
                      sig: np.ndarray, shift1, params: dict) -> np.ndarray:
    """Apply one `tpo_*` filter to the executor's `mask` array.

    The executor passes its `_shift1` helper as `shift1` so we get
    the same no-lookahead semantics. `params` is the filter dict
    from the spec (so `tpo_value_rejection` can read e.g.
    `params.get("min_close_pct", ...)`).

    Returns the updated mask.

    No-lookahead policy
    -------------------
    All TPO levels come from the PREVIOUS session, so they are
    always known at H4[i] without referencing any future bar. We
    therefore do NOT shift them by 1 again — but we DO shift any
    "current bar position vs TPO level" comparison by 1, because
    the H4 bar's close is unknown at entry time. (See vwap_dist
    fix in Batch F for the same pattern.)
    """
    needed = ("prev_session_tpo_poc", "prev_session_tpo_vah",
              "prev_session_tpo_val")
    if not _have_columns(h4, *needed):
        # Refuse silently allowing trades through without TPO data.
        return mask & False

    poc = h4["prev_session_tpo_poc"].values
    vah = h4["prev_session_tpo_vah"].values
    val = h4["prev_session_tpo_val"].values
    c = h4["close"].values
    prev_close = shift1(c)

    valid = np.isfinite(poc) & np.isfinite(vah) & np.isfinite(val)

    if token == "tpo_poc":
        # Allow trades when prev close is within `tol_atr` ATR of POC.
        # Default: just require POC defined (no positional constraint).
        return mask & valid
    if token == "tpo_vah":
        # Trades that prefer to be near VAH at entry (rejection setups)
        return mask & valid & (prev_close >= val) & (prev_close <= vah)
    if token == "tpo_val":
        return mask & valid & (prev_close >= val) & (prev_close <= vah)
    if token == "tpo_value_acceptance":
        # prev close inside the [val, vah] band -> price accepts value
        inside = (prev_close >= val) & (prev_close <= vah)
        return mask & valid & inside
    if token == "tpo_value_rejection":
        # prev close OUTSIDE the value area -> failed acceptance
        outside = (prev_close < val) | (prev_close > vah)
        return mask & valid & outside
    if token == "tpo_poor_high":
        ph = h4["prev_session_tpo_poor_high"].values.astype(bool) \
            if "prev_session_tpo_poor_high" in h4.columns \
            else np.zeros_like(poc, dtype=bool)
        return mask & valid & ph
    if token == "tpo_poor_low":
        pl = h4["prev_session_tpo_poor_low"].values.astype(bool) \
            if "prev_session_tpo_poor_low" in h4.columns \
            else np.zeros_like(poc, dtype=bool)
        return mask & valid & pl
    if token == "tpo_excess":
        eh = h4["prev_session_tpo_excess_high"].values.astype(bool) \
            if "prev_session_tpo_excess_high" in h4.columns \
            else np.zeros_like(poc, dtype=bool)
        el = h4["prev_session_tpo_excess_low"].values.astype(bool) \
            if "prev_session_tpo_excess_low" in h4.columns \
            else np.zeros_like(poc, dtype=bool)
        return mask & valid & (eh | el)
    if token == "tpo_single_print":
        # rough proxy: an excess flag means a single print AT the
        # session extreme. Without per-bin counts on H4 we fall
        # back to the same condition.
        return apply_tpo_filter("tpo_excess", h4, mask, sig, shift1, params)
    raise ValueError(f"unknown TPO token: {token}")
