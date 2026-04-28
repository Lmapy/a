"""Certification ladder + failure-reason taxonomy.

The pre-Batch-F certifier returned a binary (`certified: bool`). That
collapses too many distinctions: a strategy that fails on data
availability is different from one that fails on walk-forward, which
is different from one that's WATCHLIST-grade but missing prop
robustness.

This module provides:

  CertificationLevel    8-level ordered enum, lowest = worst
  FailureReason         structured taxonomy of why a candidate failed
  CertificationVerdict  level + ordered failure reasons + detail dict
  rank_level            integer rank for sorting / leaderboard ordering
  promote / demote      utility transitions

A candidate's certification level is the HIGHEST level it can reach
given its failure reasons. Failure reasons that gate at PROP_CANDIDATE
keep the candidate at CANDIDATE; reasons that gate earlier (e.g.
`REJECTED_UNAVAILABLE_DATA`) drop it to that floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


# ---- certification ladder ---------------------------------------------------

class CertificationLevel(str, Enum):
    """Ordered worst -> best.

    `(str, Enum)` so values JSON-serialise as the level name without
    custom encoders.
    """
    REJECTED_UNAVAILABLE_DATA = "rejected_unavailable_data"
    REJECTED_BROKEN = "rejected_broken"
    RESEARCH_ONLY = "research_only"
    WATCHLIST = "watchlist"
    CANDIDATE = "candidate"
    PROP_CANDIDATE = "prop_candidate"
    CERTIFIED = "certified"
    RETIRED = "retired"


# Canonical ordering: rank goes "worst" -> "best", with RETIRED parked at
# the end as a sticky-archived state (used to be CERTIFIED but no longer
# qualifies after data/rule changes).
_LEVEL_RANK = {
    CertificationLevel.REJECTED_UNAVAILABLE_DATA: 0,
    CertificationLevel.REJECTED_BROKEN: 1,
    CertificationLevel.RESEARCH_ONLY: 2,
    CertificationLevel.WATCHLIST: 3,
    CertificationLevel.CANDIDATE: 4,
    CertificationLevel.PROP_CANDIDATE: 5,
    CertificationLevel.CERTIFIED: 6,
    CertificationLevel.RETIRED: 7,
}


def rank_level(level: CertificationLevel) -> int:
    """Integer rank — useful for sorting leaderboard rows."""
    return _LEVEL_RANK[level]


# ---- failure-reason taxonomy ------------------------------------------------

class FailureReason(str, Enum):
    """Structured, machine-readable reason why a candidate failed a gate.

    A candidate may carry many of these. The certifier computes the
    cert level from the FULL set; the report aggregates them into
    histograms so you can answer "what's the most common gate failure
    in this run?"
    """
    # --- data / capability ---
    REJECTED_UNAVAILABLE_DATA = "rejected_unavailable_data"
    FAIL_LOOKAHEAD = "fail_lookahead"

    # --- structural / executor ---
    FAIL_SPREAD_STRESS = "fail_spread_stress"
    FAIL_RESOLUTION_LIMITED = "fail_resolution_limited"

    # --- statistical robustness ---
    FAIL_WALK_FORWARD = "fail_walk_forward"
    FAIL_HOLDOUT = "fail_holdout"
    FAIL_RANDOM_BASELINE = "fail_random_baseline"
    FAIL_LABEL_PERMUTATION = "fail_label_permutation"
    FAIL_BLOCK_BOOTSTRAP = "fail_block_bootstrap"
    FAIL_YEARLY_CONSISTENCY = "fail_yearly_consistency"
    FAIL_OVERFITTING_RISK = "fail_overfitting_risk"
    FAIL_ONE_YEAR_DEPENDENCY = "fail_one_year_dependency"
    FAIL_ONE_SESSION_DEPENDENCY = "fail_one_session_dependency"

    # --- trade-shape / executability ---
    FAIL_TOO_MANY_TRADES = "fail_too_many_trades"
    FAIL_TOO_FEW_TRADES = "fail_too_few_trades"
    FAIL_PROFIT_FACTOR = "fail_profit_factor"
    FAIL_BIGGEST_TRADE_SHARE = "fail_biggest_trade_share"
    FAIL_MAX_DRAWDOWN = "fail_max_drawdown"

    # --- prop-firm specific ---
    FAIL_DAILY_LOSS_LIMIT = "fail_daily_loss_limit"
    FAIL_TRAILING_DRAWDOWN = "fail_trailing_drawdown"
    FAIL_LOW_PASS_PROBABILITY = "fail_low_pass_probability"
    FAIL_HIGH_BLOWUP_PROBABILITY = "fail_high_blowup_probability"
    FAIL_POOR_PAYOUT_SURVIVAL = "fail_poor_payout_survival"
    FAIL_ACCOUNT_UNVERIFIED = "fail_account_unverified"


# Failure reasons that, if present, cap the candidate at a particular
# level (and prevent promotion above it). A candidate with both a
# data-rejection reason and a walk-forward fail still ends up
# REJECTED_UNAVAILABLE_DATA — the lowest cap wins.
_FAILURE_CAP: dict[FailureReason, CertificationLevel] = {
    FailureReason.REJECTED_UNAVAILABLE_DATA: CertificationLevel.REJECTED_UNAVAILABLE_DATA,
    FailureReason.FAIL_LOOKAHEAD: CertificationLevel.REJECTED_BROKEN,

    # Statistical / executor failures cap at RESEARCH_ONLY (interesting
    # but does not survive validation).
    FailureReason.FAIL_WALK_FORWARD: CertificationLevel.RESEARCH_ONLY,
    FailureReason.FAIL_HOLDOUT: CertificationLevel.RESEARCH_ONLY,
    FailureReason.FAIL_RANDOM_BASELINE: CertificationLevel.RESEARCH_ONLY,
    FailureReason.FAIL_LABEL_PERMUTATION: CertificationLevel.RESEARCH_ONLY,
    FailureReason.FAIL_BLOCK_BOOTSTRAP: CertificationLevel.RESEARCH_ONLY,
    FailureReason.FAIL_YEARLY_CONSISTENCY: CertificationLevel.WATCHLIST,
    FailureReason.FAIL_ONE_YEAR_DEPENDENCY: CertificationLevel.WATCHLIST,
    FailureReason.FAIL_ONE_SESSION_DEPENDENCY: CertificationLevel.WATCHLIST,
    FailureReason.FAIL_OVERFITTING_RISK: CertificationLevel.RESEARCH_ONLY,
    FailureReason.FAIL_SPREAD_STRESS: CertificationLevel.RESEARCH_ONLY,
    FailureReason.FAIL_RESOLUTION_LIMITED: CertificationLevel.REJECTED_BROKEN,

    # Trade-shape / metric failures cap at CANDIDATE (passes basic
    # validation but isn't a prop candidate).
    FailureReason.FAIL_TOO_MANY_TRADES: CertificationLevel.CANDIDATE,
    FailureReason.FAIL_TOO_FEW_TRADES: CertificationLevel.CANDIDATE,
    FailureReason.FAIL_PROFIT_FACTOR: CertificationLevel.CANDIDATE,
    FailureReason.FAIL_BIGGEST_TRADE_SHARE: CertificationLevel.CANDIDATE,
    FailureReason.FAIL_MAX_DRAWDOWN: CertificationLevel.CANDIDATE,

    # Prop-firm-specific failures cap at PROP_CANDIDATE (prop sim
    # surfaced something; cannot certify).
    FailureReason.FAIL_DAILY_LOSS_LIMIT: CertificationLevel.PROP_CANDIDATE,
    FailureReason.FAIL_TRAILING_DRAWDOWN: CertificationLevel.PROP_CANDIDATE,
    FailureReason.FAIL_LOW_PASS_PROBABILITY: CertificationLevel.PROP_CANDIDATE,
    FailureReason.FAIL_HIGH_BLOWUP_PROBABILITY: CertificationLevel.PROP_CANDIDATE,
    FailureReason.FAIL_POOR_PAYOUT_SURVIVAL: CertificationLevel.PROP_CANDIDATE,
    FailureReason.FAIL_ACCOUNT_UNVERIFIED: CertificationLevel.PROP_CANDIDATE,
}


def cap_for(reason: FailureReason) -> CertificationLevel:
    """The highest level a candidate carrying this reason can reach."""
    return _FAILURE_CAP[reason]


# ---- verdict object ---------------------------------------------------------

@dataclass
class CertificationVerdict:
    level: CertificationLevel
    failure_reasons: list[FailureReason] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "level": self.level.value,
            "failure_reasons": [r.value for r in self.failure_reasons],
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "CertificationVerdict":
        return cls(
            level=CertificationLevel(payload["level"]),
            failure_reasons=[FailureReason(r) for r in payload.get("failure_reasons", [])],
            detail=dict(payload.get("detail", {})),
        )


def verdict_from_reasons(reasons: Iterable[FailureReason],
                          *,
                          best_possible: CertificationLevel = CertificationLevel.CERTIFIED,
                          detail: dict | None = None,
                          ) -> CertificationVerdict:
    """Compute the certification level given a set of failure reasons.

    The level is `min(best_possible, min(cap_for(r) for r in reasons))`,
    where `min` is taken on the integer rank.
    """
    reasons = list(reasons)
    if not reasons:
        return CertificationVerdict(level=best_possible, detail=dict(detail or {}))
    caps = [cap_for(r) for r in reasons]
    lowest = min([best_possible] + caps, key=rank_level)
    return CertificationVerdict(level=lowest, failure_reasons=reasons,
                                 detail=dict(detail or {}))


def promote(level: CertificationLevel) -> CertificationLevel:
    """Move one level up the ladder; CERTIFIED is the ceiling, RETIRED is sticky."""
    if level in (CertificationLevel.CERTIFIED, CertificationLevel.RETIRED):
        return level
    rank = rank_level(level)
    nxt = next((lv for lv, r in _LEVEL_RANK.items()
                if r == rank + 1 and lv != CertificationLevel.RETIRED), level)
    return nxt


def demote(level: CertificationLevel) -> CertificationLevel:
    """Move one level down. Floor is REJECTED_UNAVAILABLE_DATA."""
    if level == CertificationLevel.REJECTED_UNAVAILABLE_DATA:
        return level
    rank = rank_level(level)
    prv = next((lv for lv, r in _LEVEL_RANK.items() if r == rank - 1), level)
    return prv
