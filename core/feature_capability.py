"""Feature capability system — what data the harness actually has.

The hardened harness is OHLC-only on Dukascopy candles. There is NO
real traded volume, NO bid/ask volume, NO footprint, NO delta, NO CVD,
NO order book. Strategies that depend on those features must be
rejected loudly; they cannot be backed by the data on disk.

This module provides:

  CapabilityRegistry       what features are / are not available
  STRATEGY_FEATURE_TAGS    what data a feature requires
  classify_candidate       returns "ok" or "rejected_unavailable_data"
                            with the list of unavailable features
  reject_unavailable       mutates a PropCandidate (or dict) with the
                            REJECTED_UNAVAILABLE_DATA certification
                            level and structured failure reasons

A strategy's `signal`, every entry in `filters`, and the `entry`,
`stop`, `exit` blocks are scanned for a `type` token. The token is
looked up in `STRATEGY_FEATURE_TAGS`; tags with `requires_real_volume`,
`requires_bid_ask_volume`, etc. are rejected unless the registry
says that capability is available.

The registry CAN be expanded later (e.g. a Dukascopy backfill that
includes tick volume). For now everything volume-flavoured is OFF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# ---- the canonical capability registry --------------------------------------

@dataclass(frozen=True)
class CapabilityRegistry:
    """What the harness can actually compute from the data on disk.

    Set frozen=True so a runtime mutation cannot quietly enable a
    feature the data doesn't support.
    """
    ohlc: bool = True              # H4 / M15 / M5 / M1 OHLC on disk
    spread_mean: bool = True       # ask-bid mean per bar from Dukascopy ticks
    tick_count: bool = True        # number of ticks per bar (per-bar count, not directional)
    tpo: bool = True               # time-at-price, derivable from OHLC alone

    real_volume: bool = False      # actual traded volume — not on disk
    bid_ask_volume: bool = False   # buyer/seller side volume — not on disk
    footprint: bool = False        # per-price aggressor side — not on disk
    delta: bool = False            # signed flow per bar — not on disk
    cvd: bool = False              # cumulative volume delta — not on disk
    orderbook: bool = False        # DOM snapshots — not on disk
    vwap: bool = False             # requires real volume at each price
    volume_profile: bool = False   # requires real volume at each price


CANONICAL_REGISTRY = CapabilityRegistry()


# ---- feature requirements per strategy token --------------------------------
#
# Each key is a `type` token that may appear in a candidate's signal /
# filters / entry / stop / exit. The value is the list of capabilities
# that token requires. A token whose requirements aren't all available
# in the registry is rejected.
#
# Tokens NOT listed here are assumed OHLC-only and require no
# additional capability beyond `ohlc`.

STRATEGY_FEATURE_TAGS: dict[str, list[str]] = {
    # ---- already-implemented OHLC tokens (allowed) ----
    "prev_color": [],
    "prev_color_inverse": [],
    "displacement": [],
    "sweep_rejection": [],
    "failed_continuation": [],
    "multi_bar_directional": [],
    "body_atr": [],
    "session": [],
    "regime": [],
    "min_streak": [],
    "atr_percentile": [],
    "pdh_pdl": [],
    "regime_class": [],
    "wick_ratio": [],
    "h4_close": [],
    "h4_open": [],
    "h4_atr": [],
    "m15_atr": [],
    "prev_h4_open": [],
    "prev_h4_extreme": [],
    "prev_h4_extreme_tp": [],
    "touch_entry": [],
    "fib_limit_entry": [],
    "reaction_close": [],
    "zone_midpoint_limit": [],
    "sweep_reclaim": [],
    "minor_structure_break": [],
    "delayed_entry_1": [],
    "delayed_entry_2": [],
    "m15_open": [],
    "m15_confirm": [],
    "m15_atr_stop": [],
    "m15_retrace_50": [],
    "m15_retrace_fib": [],

    # ---- NEW OHLC-only tokens (allowed; for Batch G families) ----
    "session_twap": [],
    "anchored_mean": [],
    "session_mean_price": [],
    "typical_price_mean": [],
    "fair_value_proxy": [],
    "atr_distance_from_session_mean": [],
    "tpo_poc": ["tpo"],
    "tpo_vah": ["tpo"],
    "tpo_val": ["tpo"],
    "tpo_value_area": ["tpo"],
    "tpo_single_print": ["tpo"],
    "tpo_poor_high": ["tpo"],
    "tpo_poor_low": ["tpo"],
    "tpo_excess": ["tpo"],
    "tpo_value_acceptance": ["tpo"],
    "tpo_value_rejection": ["tpo"],
    "initial_balance_high": [],
    "initial_balance_low": [],
    "initial_balance_breakout": [],
    "initial_balance_failure": [],
    "opening_range_breakout": [],
    "opening_range_failed_breakout": [],
    "opening_range_retest": [],
    "previous_session_high": [],
    "previous_session_low": [],
    "previous_session_midpoint": [],
    "previous_h4_midpoint": [],
    "compression_breakout": [],
    "failed_breakout_reversal": [],
    "atr_extension_reclaim": [],

    # ---- REJECTED tokens (require unavailable real data) ----
    "vwap": ["vwap"],
    "vwap_dist": ["vwap"],
    "htf_vwap_dist": ["vwap"],
    "anchored_vwap": ["vwap"],
    "session_vwap": ["vwap"],
    "vwap_band": ["vwap"],
    "volume_profile": ["volume_profile"],
    "volume_poc": ["volume_profile"],
    "volume_vah": ["volume_profile"],
    "volume_val": ["volume_profile"],
    "hvn": ["volume_profile"],
    "lvn": ["volume_profile"],
    "volume_node_rejection": ["volume_profile"],
    "footprint": ["footprint"],
    "footprint_imbalance": ["footprint"],
    "delta": ["delta"],
    "delta_divergence": ["delta"],
    "cvd": ["cvd"],
    "cvd_divergence": ["cvd"],
    "bid_ask_imbalance": ["bid_ask_volume"],
    "order_flow_imbalance": ["bid_ask_volume"],
    "aggressive_buyer": ["bid_ask_volume"],
    "aggressive_seller": ["bid_ask_volume"],
    "dom_liquidity": ["orderbook"],
    "orderbook_imbalance": ["orderbook"],
    "iceberg": ["orderbook"],
}


# ---- VWAP / Volume Profile rename hints -------------------------------------
#
# When a user proposes a strategy that mentions VWAP or Volume Profile
# the proposer should convert it to an OHLC-only equivalent or reject
# it. These two maps document the suggested replacements; they are
# advisory, not enforced.

VWAP_REPLACEMENTS: dict[str, str] = {
    "vwap": "session_twap",
    "anchored_vwap": "anchored_mean",
    "session_vwap": "session_mean_price",
    "vwap_dist": "atr_distance_from_session_mean",
    "htf_vwap_dist": "atr_distance_from_session_mean",
    "vwap_band": "atr_distance_from_session_mean",
}

VOLUME_PROFILE_REPLACEMENTS: dict[str, str] = {
    "volume_poc": "tpo_poc",
    "volume_vah": "tpo_vah",
    "volume_val": "tpo_val",
    "hvn": "tpo_value_acceptance",
    "lvn": "tpo_single_print",
    "volume_node_rejection": "tpo_value_rejection",
    "volume_profile": "tpo_value_area",
}


# ---- classification ---------------------------------------------------------

@dataclass
class CapabilityVerdict:
    status: str                         # "ok" | "rejected_unavailable_data" | "unknown_token"
    unavailable_features: list[str] = field(default_factory=list)
    unavailable_tokens: list[str] = field(default_factory=list)
    unknown_tokens: list[str] = field(default_factory=list)
    rename_hints: dict[str, str] = field(default_factory=dict)


def _iter_tokens(spec_or_candidate) -> list[str]:
    """Pull every `type` token out of a Spec/dict/PropCandidate.

    Accepts either:
      * a plain dict that has signal/filters/entry/stop/exit keys
      * an object with the same attribute names (Spec / PropCandidate)
    """
    g = (lambda k: spec_or_candidate.get(k)
         if isinstance(spec_or_candidate, dict)
         else getattr(spec_or_candidate, k, None))
    out: list[str] = []
    for k in ("signal", "entry", "stop", "exit"):
        block = g(k)
        if isinstance(block, dict) and "type" in block:
            out.append(str(block["type"]))
    filters = g("filters") or []
    for f in filters:
        if isinstance(f, dict) and "type" in f:
            out.append(str(f["type"]))
    return out


def classify_candidate(spec_or_candidate,
                        registry: CapabilityRegistry = CANONICAL_REGISTRY,
                        ) -> CapabilityVerdict:
    """Decide whether the candidate can run on the available data.

    Unknown tokens (not in `STRATEGY_FEATURE_TAGS`) are reported as
    `unknown_tokens` and treated as OHLC-only by default — strategy
    proposers should add new tokens to the tag map; the existing
    candidate paths are not penalised by missing entries.
    """
    tokens = _iter_tokens(spec_or_candidate)
    unavail_feats: set[str] = set()
    unavail_tok: list[str] = []
    unknown: list[str] = []
    rename_hints: dict[str, str] = {}
    for t in tokens:
        reqs = STRATEGY_FEATURE_TAGS.get(t)
        if reqs is None:
            unknown.append(t)
            continue
        missing = [r for r in reqs if not getattr(registry, r, False)]
        if missing:
            unavail_feats.update(missing)
            unavail_tok.append(t)
            if t in VWAP_REPLACEMENTS:
                rename_hints[t] = VWAP_REPLACEMENTS[t]
            elif t in VOLUME_PROFILE_REPLACEMENTS:
                rename_hints[t] = VOLUME_PROFILE_REPLACEMENTS[t]
    status = "rejected_unavailable_data" if unavail_feats else "ok"
    return CapabilityVerdict(
        status=status,
        unavailable_features=sorted(unavail_feats),
        unavailable_tokens=unavail_tok,
        unknown_tokens=unknown,
        rename_hints=rename_hints,
    )


def is_available(token: str,
                 registry: CapabilityRegistry = CANONICAL_REGISTRY) -> bool:
    """Convenience: is this single feature token available right now?"""
    reqs = STRATEGY_FEATURE_TAGS.get(token)
    if reqs is None:
        return True   # unknown token -> assumed OHLC-only
    return all(getattr(registry, r, False) for r in reqs)


def assert_only_ohlc_only(spec_or_candidate,
                           registry: CapabilityRegistry = CANONICAL_REGISTRY) -> None:
    """Hard assertion for runners that must not silently accept
    unavailable-data candidates."""
    v = classify_candidate(spec_or_candidate, registry=registry)
    if v.status != "ok":
        raise ValueError(
            f"candidate uses unavailable data features "
            f"{v.unavailable_features}; offending tokens "
            f"{v.unavailable_tokens}; suggested renames {v.rename_hints}")
