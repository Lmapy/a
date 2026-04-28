"""Entry-model ↔ entry-timeframe compatibility map.

Not every entry model is meaningful on every timeframe. Sweep-and-reclaim
needs ticks fast enough to register the sweep; structural breaks need
enough sub-bars per parent candle for "structure" to exist; touch /
fib / zone entries are coarser and work on M5+.

If a model is requested outside its compatible timeframes the runner
must skip the spec by default (or, with `allow_resolution_limited`,
mark it as `resolution_limited` and refuse to certify it as
high-confidence).
"""
from __future__ import annotations

ENTRY_MODEL_TIMEFRAME_MAP: dict[str, list[str]] = {
    "touch_entry":            ["M5", "M15", "M30"],
    "reaction_close":         ["M5", "M15", "M30"],
    "sweep_reclaim":          ["M1", "M3", "M5"],
    "minor_structure_break":  ["M1", "M3", "M5"],
    "delayed_entry_1":        ["M5", "M15"],
    "delayed_entry_2":        ["M5", "M15"],
    "fib_limit_entry":        ["M5", "M15", "M30"],
    "zone_midpoint_limit":    ["M5", "M15", "M30"],
}


def is_compatible(entry_model: str, entry_timeframe: str) -> bool:
    return entry_timeframe in ENTRY_MODEL_TIMEFRAME_MAP.get(entry_model, [])


def compatibility_status(entry_model: str, entry_timeframe: str,
                         data_available: bool) -> str:
    """Return one of: ok | data_unavailable | resolution_limited | unknown_model."""
    if entry_model not in ENTRY_MODEL_TIMEFRAME_MAP:
        return "unknown_model"
    if not data_available:
        return "data_unavailable"
    if not is_compatible(entry_model, entry_timeframe):
        return "resolution_limited"
    return "ok"
