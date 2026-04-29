"""Strategy family generators (Batch G).

Ten OHLC-only families, each emitting a small deterministic batch of
`PropCandidate` objects. The grid is intentionally controlled — each
family produces a handful of variants, not an exhaustive sweep. The
entry / risk / daily-rule labs (separate modules) take a base
candidate and multiply variants on those axes; the grid composer
(`strategies.grid`) decides how those layers combine.

All families consume only OHLC-derivable features. The two TPO
families (`tpo_value_rejection`, `tpo_poc_reversion`) emit candidates
whose `filters` reference TPO tokens (`tpo_value_acceptance`,
`tpo_poor_high`, etc.) — these are accepted by the Batch F feature-
capability gate but require the executor's TPO filter
implementations to be wired in Batch H. Until that happens the
orchestrator reports them as `data_unavailable` for the executor;
the candidates are well-formed and round-trip through JSON.

Family registry
---------------
The module-level dict `FAMILIES` maps family-id -> generator. The
`all_candidates()` helper concatenates every family's output for the
tier-1 grid.

Family naming
-------------
Family IDs follow the user's brief verbatim:

  session_sweep_reclaim
  opening_range_breakout_retest
  opening_range_failed_breakout
  previous_h4_range_retracement
  previous_h4_sweep_reclaim
  tpo_value_rejection
  tpo_poc_reversion
  atr_extension_reclaim
  compression_breakout
  failed_breakout_reversal
"""
from __future__ import annotations

from typing import Callable, Iterable

from core.candidate import (
    AccountRef, DailyRulesBlock, PropCandidate, RiskBlock,
)


FamilyGenerator = Callable[..., list[PropCandidate]]


# ---- session helpers -------------------------------------------------------

# UTC hour ranges for each major session. Used as `session.hours_utc`
# values in candidate filters.
SESSION_LONDON_HOURS = [7, 8, 9, 10, 11]
SESSION_NY_HOURS = [12, 13, 14, 15, 16]
SESSION_LONDON_OPEN = [7, 8]
SESSION_NY_OPEN = [12, 13]

# default account: smallest verified prop account so we get a tighter
# blow-up gate. The risk sweep and prop simulator can swap account
# at runtime.
_DEFAULT_ACCOUNT = "topstep_50k"


def _account(name: str = _DEFAULT_ACCOUNT) -> AccountRef:
    return AccountRef(name=name, instrument="MGC")


def _id(family: str, *parts: str) -> str:
    """Compose a deterministic candidate id from family + variant tags."""
    tail = "__".join(p for p in parts if p)
    return f"{family}__{tail}" if tail else family


# ---- 1. session_sweep_reclaim ----------------------------------------------

def session_sweep_reclaim() -> list[PropCandidate]:
    """Sweep the prior session high/low, fail/reclaim, fade.

    Uses the existing `sweep_rejection` signal. Variants over session
    filter (London-only, NY-only) and stop choice. Entry: reaction
    close (waits for a confirmation candle).
    """
    family = "session_sweep_reclaim"
    out: list[PropCandidate] = []
    for session_id, hours in (("ny", SESSION_NY_HOURS),
                                ("london", SESSION_LONDON_HOURS)):
        for stop_id, stop in (("prev_h4_extreme", {"type": "prev_h4_extreme"}),
                                ("h4_atr1", {"type": "h4_atr", "mult": 1.0,
                                              "atr_n": 14})):
            out.append(PropCandidate(
                id=_id(family, session_id, stop_id),
                family=family,
                signal={"type": "sweep_rejection"},
                filters=[
                    {"type": "session", "hours_utc": hours},
                    {"type": "regime", "ma_n": 50, "side": "against"},
                ],
                entry={"type": "reaction_close"},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes="sweep & reclaim of prior session extreme; fade",
            ))
    return out


# ---- 2. opening_range_breakout_retest --------------------------------------

def opening_range_breakout_retest() -> list[PropCandidate]:
    """Break above (or below) the opening-range extreme; enter on retest.

    Uses `displacement` for the directional break with a strong-close
    requirement. The `pdh_pdl` filter in `breakout` mode keeps the
    trade aligned with the prior-day-range break. Entry: minor
    structure break, which waits for the M15 micro-pullback to fail.
    """
    family = "opening_range_breakout_retest"
    out: list[PropCandidate] = []
    for session_id, hours in (("ny_open", SESSION_NY_OPEN),
                                ("london_open", SESSION_LONDON_OPEN)):
        for stop_id, stop in (("prev_h4_open", {"type": "prev_h4_open"}),
                                ("h4_atr1", {"type": "h4_atr", "mult": 1.0,
                                              "atr_n": 14})):
            out.append(PropCandidate(
                id=_id(family, session_id, stop_id),
                family=family,
                signal={"type": "displacement", "min_body_atr": 0.7,
                         "atr_n": 14, "close_pct": 0.30},
                filters=[
                    {"type": "session", "hours_utc": hours},
                    {"type": "pdh_pdl", "mode": "breakout"},
                ],
                entry={"type": "minor_structure_break"},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes="opening range breakout, enter on retest/structure break",
            ))
    return out


# ---- 3. opening_range_failed_breakout --------------------------------------

def opening_range_failed_breakout() -> list[PropCandidate]:
    """The break above PDH (or below PDL) FAILS — fade back inside.

    Uses `failed_continuation` for the signal: two prior bars same
    direction but the latest closes against. The `pdh_pdl inside`
    filter requires price to have returned inside the prior-day
    range, isolating the failed-breakout pattern.
    """
    family = "opening_range_failed_breakout"
    out: list[PropCandidate] = []
    for session_id, hours in (("ny_open", SESSION_NY_OPEN),
                                ("london_open", SESSION_LONDON_OPEN)):
        for stop_id, stop in (("prev_h4_extreme", {"type": "prev_h4_extreme"}),
                                ("h4_atr1", {"type": "h4_atr", "mult": 1.0,
                                              "atr_n": 14})):
            out.append(PropCandidate(
                id=_id(family, session_id, stop_id),
                family=family,
                signal={"type": "failed_continuation"},
                filters=[
                    {"type": "session", "hours_utc": hours},
                    {"type": "pdh_pdl", "mode": "inside"},
                ],
                entry={"type": "reaction_close"},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes="failed opening-range breakout reversal",
            ))
    return out


# ---- 4. previous_h4_range_retracement --------------------------------------

def previous_h4_range_retracement() -> list[PropCandidate]:
    """Continuation entry on a fib retrace inside the prior-H4 range.

    Uses `prev_color` for direction and the M15 fib limit entry to
    enter at a deeper retracement of the prior H4 candle's body.
    Filter: body_atr >= 0.5 to require a meaningful prior bar.
    """
    family = "previous_h4_range_retracement"
    out: list[PropCandidate] = []
    for level in (0.382, 0.5, 0.618):
        for stop_id, stop in (("prev_h4_open", {"type": "prev_h4_open"}),
                                ("prev_h4_extreme", {"type": "prev_h4_extreme"})):
            out.append(PropCandidate(
                id=_id(family, f"fib{level}", stop_id),
                family=family,
                signal={"type": "prev_color"},
                filters=[
                    {"type": "body_atr", "min": 0.5, "atr_n": 14},
                    {"type": "regime", "ma_n": 50, "side": "with"},
                ],
                entry={"type": "fib_limit_entry", "level": level},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes=f"prior H4 retrace continuation at fib {level}",
            ))
    return out


# ---- 5. previous_h4_sweep_reclaim ------------------------------------------

def previous_h4_sweep_reclaim() -> list[PropCandidate]:
    """The current H4 sweeps the prior H4 extreme then closes back inside.

    Uses `sweep_rejection` signal (same as session_sweep_reclaim) but
    with a different filter set tuned to per-bar reclaims rather
    than session-extreme reclaims. Entry: minor structure break on
    M15 keeps the timing tight."""
    family = "previous_h4_sweep_reclaim"
    out: list[PropCandidate] = []
    for streak_id, k in (("k1", 1), ("k2", 2)):
        for stop_id, stop in (("prev_h4_open", {"type": "prev_h4_open"}),
                                ("prev_h4_extreme", {"type": "prev_h4_extreme"})):
            out.append(PropCandidate(
                id=_id(family, streak_id, stop_id),
                family=family,
                signal={"type": "sweep_rejection"},
                filters=[
                    {"type": "min_streak", "k": k},
                    {"type": "wick_ratio", "min": 0.4},
                ],
                entry={"type": "minor_structure_break"},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes="prior H4 sweep & reclaim",
            ))
    return out


# ---- 6. tpo_value_rejection ------------------------------------------------

def tpo_value_rejection() -> list[PropCandidate]:
    """Failed acceptance of yesterday's TPO value area.

    Open outside prior value -> probe back inside -> fail to accept
    -> trade back toward the POC. Requires TPO levels per session;
    the executor wires those in Batch H. Until then these candidates
    will be reported as `requires_executor_extension=tpo_filter` and
    skipped.
    """
    family = "tpo_value_rejection"
    out: list[PropCandidate] = []
    for session_id, hours in (("ny_open", SESSION_NY_OPEN),
                                ("london_open", SESSION_LONDON_OPEN)):
        for stop_id, stop in (("prev_h4_extreme", {"type": "prev_h4_extreme"}),
                                ("h4_atr1", {"type": "h4_atr", "mult": 1.0,
                                              "atr_n": 14})):
            out.append(PropCandidate(
                id=_id(family, session_id, stop_id),
                family=family,
                signal={"type": "failed_continuation"},
                filters=[
                    {"type": "session", "hours_utc": hours},
                    {"type": "tpo_value_rejection"},
                ],
                entry={"type": "reaction_close"},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes="failed acceptance of prior session TPO value area",
                provenance={"requires_executor_extension": "tpo_filter"},
            ))
    return out


# ---- 7. tpo_poc_reversion --------------------------------------------------

def tpo_poc_reversion() -> list[PropCandidate]:
    """Mean-revert toward yesterday's TPO POC.

    Setup: price extends >= N ATR away from prior session TPO POC,
    fade back. Same Batch H requirement as `tpo_value_rejection`.
    """
    family = "tpo_poc_reversion"
    out: list[PropCandidate] = []
    for session_id, hours in (("ny", SESSION_NY_HOURS),
                                ("london", SESSION_LONDON_HOURS)):
        for atr_id, atr_n in (("atr14", 14), ("atr20", 20)):
            out.append(PropCandidate(
                id=_id(family, session_id, atr_id),
                family=family,
                signal={"type": "prev_color_inverse"},
                filters=[
                    {"type": "session", "hours_utc": hours},
                    {"type": "regime", "ma_n": 50, "side": "against"},
                    {"type": "tpo_poc"},
                ],
                entry={"type": "fib_limit_entry", "level": 0.5},
                stop={"type": "h4_atr", "mult": 1.5, "atr_n": atr_n},
                exit={"type": "prev_h4_extreme_tp"},
                account=_account(),
                notes="mean-revert to prior session TPO POC",
                provenance={"requires_executor_extension": "tpo_filter"},
            ))
    return out


# ---- 8. atr_extension_reclaim ----------------------------------------------

def atr_extension_reclaim() -> list[PropCandidate]:
    """Price extends >= N ATR from prior bar's mid; sweep & reclaim
    confirms reversion.

    Uses `prev_color_inverse` to express the fade direction and an
    ATR-percentile filter to enforce the "extension" condition (top
    decile of ATR). Stop = M15 ATR.
    """
    family = "atr_extension_reclaim"
    out: list[PropCandidate] = []
    for atr_id, atr_n in (("atr14", 14), ("atr20", 20)):
        for stop_mult_id, mult in (("m1", 1.0), ("m1_5", 1.5)):
            out.append(PropCandidate(
                id=_id(family, atr_id, stop_mult_id),
                family=family,
                signal={"type": "prev_color_inverse"},
                filters=[
                    {"type": "atr_percentile", "window": 100, "atr_n": atr_n,
                     "lo": 0.80, "hi": 1.0},
                    {"type": "regime_class", "allow": ["expansion"],
                     "atr_n": atr_n},
                ],
                entry={"type": "reaction_close"},
                stop={"type": "m15_atr", "mult": mult, "atr_n": atr_n},
                exit={"type": "h4_close"},
                account=_account(),
                notes="ATR-extension reversion fade",
            ))
    return out


# ---- 9. compression_breakout ------------------------------------------------

def compression_breakout() -> list[PropCandidate]:
    """After low-ATR-percentile compression, an ATR-expansion bar
    breaks out; trade continuation.
    """
    family = "compression_breakout"
    out: list[PropCandidate] = []
    for atr_id, atr_n in (("atr14", 14), ("atr20", 20)):
        for stop_id, stop in (("prev_h4_open", {"type": "prev_h4_open"}),
                                ("h4_atr1", {"type": "h4_atr", "mult": 1.0,
                                              "atr_n": 14})):
            out.append(PropCandidate(
                id=_id(family, atr_id, stop_id),
                family=family,
                signal={"type": "displacement", "min_body_atr": 1.0,
                         "atr_n": atr_n, "close_pct": 0.25},
                filters=[
                    {"type": "atr_percentile", "window": 100, "atr_n": atr_n,
                     "lo": 0.0, "hi": 0.30},
                    {"type": "regime", "ma_n": 50, "side": "with"},
                ],
                entry={"type": "touch_entry"},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes="compression-then-expansion breakout continuation",
            ))
    return out


# ---- 10. failed_breakout_reversal ------------------------------------------

def failed_breakout_reversal() -> list[PropCandidate]:
    """A breakout from a tight range fails — trade against the failed
    breakout. Uses `failed_continuation` signal under a regime filter
    that says we're in expansion (so the failure is meaningful).
    """
    family = "failed_breakout_reversal"
    out: list[PropCandidate] = []
    for session_id, hours in (("ny", SESSION_NY_HOURS),
                                ("london", SESSION_LONDON_HOURS)):
        for stop_id, stop in (("prev_h4_extreme", {"type": "prev_h4_extreme"}),
                                ("h4_atr1_5", {"type": "h4_atr", "mult": 1.5,
                                                "atr_n": 14})):
            out.append(PropCandidate(
                id=_id(family, session_id, stop_id),
                family=family,
                signal={"type": "failed_continuation"},
                filters=[
                    {"type": "session", "hours_utc": hours},
                    {"type": "regime_class", "allow": ["expansion", "trend"],
                     "atr_n": 14},
                ],
                entry={"type": "reaction_close"},
                stop=stop,
                exit={"type": "h4_close"},
                account=_account(),
                notes="failed breakout reversal",
            ))
    return out


# ---- registry --------------------------------------------------------------

FAMILIES: dict[str, FamilyGenerator] = {
    "session_sweep_reclaim": session_sweep_reclaim,
    "opening_range_breakout_retest": opening_range_breakout_retest,
    "opening_range_failed_breakout": opening_range_failed_breakout,
    "previous_h4_range_retracement": previous_h4_range_retracement,
    "previous_h4_sweep_reclaim": previous_h4_sweep_reclaim,
    "tpo_value_rejection": tpo_value_rejection,
    "tpo_poc_reversion": tpo_poc_reversion,
    "atr_extension_reclaim": atr_extension_reclaim,
    "compression_breakout": compression_breakout,
    "failed_breakout_reversal": failed_breakout_reversal,
}


def all_candidates(*, families: Iterable[str] | None = None
                   ) -> list[PropCandidate]:
    """Concatenate every family's output. If `families` is given, only
    those family ids are emitted (used by the orchestrator's
    `--families` filter)."""
    out: list[PropCandidate] = []
    for fid, gen in FAMILIES.items():
        if families is not None and fid not in families:
            continue
        out.extend(gen())
    return out


def family_ids() -> list[str]:
    return list(FAMILIES.keys())
