"""Catalogue of strategy families.

Each family declares its market logic, expected failure mode, compatible
timeframes, and a parameterised spec template that the spec-builder
agent expands into concrete specs.
"""
from __future__ import annotations

from typing import Iterable

# Each family produces a list of variant overrides; the spec builder
# expands the template against them.
STRATEGY_FAMILIES: dict[str, dict] = {

    # A. Strong-body continuation (already certified family in v1).
    "strong_body_continuation": {
        "description": "Prior H4 candle body >= X * ATR(14); trade in same "
                       "direction only if regime agrees.",
        "expected_failure_mode": "Range-bound low-volatility sessions where "
                                 "strong bodies still revert.",
        "required_data": ["H4", "M15"],
        "preferred_entry_timeframes": ["M15", "M5"],
        "family_class": "continuation",
        "template": {
            "signal": {"type": "prev_color"},
            "filters": [
                {"type": "body_atr", "min": 0.5, "atr_n": 14},
                {"type": "regime", "ma_n": 50, "side": "with"},
            ],
            "entry": {"type": "touch_entry"},
            "stop": {"type": "prev_h4_open"},
            "exit": {"type": "h4_close"},
        },
        "variants": [
            {"filters[0].min": 0.5},
            {"filters[0].min": 0.7},
            {"filters[0].min": 1.0},
        ],
    },

    # B. Exhaustion reversal (NEW).
    "exhaustion_reversal": {
        "description": "Prior H4 candle has a large wick (body / range <= "
                       "wick_ratio) and closes back inside the prior bar's "
                       "range; trade reversal.",
        "expected_failure_mode": "Trend continuation days where the wick is "
                                 "just a brief retest before the trend resumes.",
        "required_data": ["H4", "M15"],
        "preferred_entry_timeframes": ["M15", "M5"],
        "family_class": "reversal",
        "template": {
            "signal": {"type": "prev_color_inverse"},  # NEW signal type
            "filters": [
                {"type": "wick_ratio", "min": 0.6},     # NEW filter type
                {"type": "regime", "ma_n": 50, "side": "against"},
            ],
            "entry": {"type": "reaction_close"},
            "stop": {"type": "prev_h4_extreme"},
            "exit": {"type": "h4_close"},
        },
        "variants": [
            {"filters[0].min": 0.6},
            {"filters[0].min": 0.75},
        ],
    },

    # C. Sweep-and-reclaim (already a registered entry model).
    "sweep_reclaim_back_to_value": {
        "description": "Previous H4 high or low is swept then reclaimed; "
                       "trade back toward value/midpoint.",
        "expected_failure_mode": "Real breakouts where the sweep is the start "
                                 "of a new leg, not a stop hunt.",
        "required_data": ["H4", "M5"],   # sweep needs sub-M15 resolution
        "preferred_entry_timeframes": ["M5", "M3"],
        "family_class": "reversal",
        "template": {
            "signal": {"type": "prev_color"},
            "filters": [],
            "entry": {"type": "sweep_reclaim"},
            "stop": {"type": "prev_h4_extreme"},
            "exit": {"type": "h4_close"},
        },
        "variants": [
            {"filters": []},
            {"filters": [{"type": "session", "hours_utc": [12, 16]}]},
        ],
    },

    # D. Fib continuation (already certified).
    "fib_continuation": {
        "description": "Trade shallow-to-deep retracement of previous H4 "
                       "candle only when trend/regime agrees.",
        "expected_failure_mode": "Deep retracements that turn into full "
                                 "reversals -- 32% of bars retrace fully.",
        "required_data": ["H4", "M15"],
        "preferred_entry_timeframes": ["M15", "M5"],
        "family_class": "continuation",
        "template": {
            "signal": {"type": "prev_color"},
            "filters": [
                {"type": "body_atr", "min": 0.5, "atr_n": 14},
                {"type": "regime", "ma_n": 50, "side": "with"},
            ],
            "entry": {"type": "fib_limit_entry", "level": 0.382},
            "stop": {"type": "prev_h4_open"},
            "exit": {"type": "h4_close"},
        },
        "variants": [
            {"entry.level": 0.382},
            {"entry.level": 0.5},
            {"entry.level": 0.618},
            {"entry.level": 0.705},
            {"entry.level": 0.786},
        ],
    },

    # E. Session expansion (NEW).
    "asia_compression_session_breakout": {
        "description": "Asia session compresses into a tight range; "
                       "London or NY session breaks the range with displacement.",
        "expected_failure_mode": "Fakeout breakouts during low-impact news; "
                                 "range expands then immediately reverses.",
        "required_data": ["H4", "M15"],
        "preferred_entry_timeframes": ["M5", "M15"],
        "family_class": "breakout",
        "template": {
            "signal": {"type": "asia_range_break"},   # NEW signal type
            "filters": [
                {"type": "session", "hours_utc": [8, 12, 16]},
            ],
            "entry": {"type": "minor_structure_break", "lookback": 3},
            "stop": {"type": "prev_h4_open"},
            "exit": {"type": "h4_close"},
        },
        "variants": [
            {"filters": [{"type": "session", "hours_utc": [8, 12, 16]}]},
            {"filters": [{"type": "session", "hours_utc": [12, 16]}]},
        ],
    },

    # F. Mean reversion to VWAP / MA (NEW).
    "vwap_mean_reversion": {
        "description": "Price extends > N std from VWAP/MA in a range "
                       "regime; fade back to mean.",
        "expected_failure_mode": "Trending regime where the extension keeps "
                                 "extending instead of reverting.",
        "required_data": ["H4", "M15"],
        "preferred_entry_timeframes": ["M15", "M5"],
        "family_class": "mean_reversion",
        "template": {
            "signal": {"type": "prev_color_inverse"},
            "filters": [
                {"type": "vwap_dist", "window": 24, "max_z": 2.5},   # require |z|>=2.5
                {"type": "regime", "ma_n": 50, "side": "against"},   # against trend = range/MR
            ],
            "entry": {"type": "reaction_close"},
            "stop": {"type": "prev_h4_extreme"},
            "exit": {"type": "h4_close"},
        },
        "variants": [
            {"filters[0].max_z": 2.0},
            {"filters[0].max_z": 2.5},
            {"filters[0].max_z": 3.0},
        ],
    },

    # G. Compression breakout (NEW).
    "compression_breakout_continuation": {
        "description": "After a low-ATR-percentile compression, an ATR "
                       "expansion bar breaks out; trade continuation.",
        "expected_failure_mode": "Late entries after the move has already "
                                 "spent most of its ATR.",
        "required_data": ["H4", "M15"],
        "preferred_entry_timeframes": ["M15", "M5"],
        "family_class": "breakout",
        "template": {
            "signal": {"type": "prev_color"},
            "filters": [
                {"type": "atr_percentile", "window": 100, "lo": 0.0, "hi": 0.30},
                {"type": "body_atr", "min": 1.0, "atr_n": 14},
            ],
            "entry": {"type": "touch_entry"},
            "stop": {"type": "prev_h4_open"},
            "exit": {"type": "h4_close"},
        },
        "variants": [
            {"filters[0].hi": 0.20},
            {"filters[0].hi": 0.30},
            {"filters[0].hi": 0.40},
        ],
    },
}


def list_families() -> list[str]:
    return list(STRATEGY_FAMILIES.keys())


def family(name: str) -> dict:
    if name not in STRATEGY_FAMILIES:
        raise KeyError(name)
    return STRATEGY_FAMILIES[name]


def all_hypotheses() -> Iterable[dict]:
    """Emit one hypothesis JSON per family (used by agent 02)."""
    for fname, f in STRATEGY_FAMILIES.items():
        yield {
            "hypothesis_id": fname,
            "market_logic": f["description"],
            "expected_failure_mode": f["expected_failure_mode"],
            "required_data": f["required_data"],
            "preferred_entry_timeframes": f["preferred_entry_timeframes"],
            "strategy_family": f["family_class"],
        }
