"""Certification ladder + failure-reason tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.certification import (
    CertificationLevel, CertificationVerdict, FailureReason,
    cap_for, demote, promote, rank_level, verdict_from_reasons,
)


def test_levels_unique_and_ordered_worst_to_best():
    levels = list(CertificationLevel)
    ranks = [rank_level(l) for l in levels]
    assert len(set(ranks)) == len(ranks), "duplicate ranks"
    # REJECTED_UNAVAILABLE_DATA is the worst, RETIRED is the sticky best
    assert min(ranks) == rank_level(CertificationLevel.REJECTED_UNAVAILABLE_DATA)
    assert max(ranks) == rank_level(CertificationLevel.RETIRED)


def test_failure_reason_caps_are_consistent():
    """Every failure reason must have a cap defined."""
    for r in FailureReason:
        cap = cap_for(r)
        assert isinstance(cap, CertificationLevel)


def test_no_reasons_means_best_possible():
    v = verdict_from_reasons([])
    assert v.level == CertificationLevel.CERTIFIED
    assert v.failure_reasons == []


def test_walk_forward_fail_caps_at_research_only():
    v = verdict_from_reasons([FailureReason.FAIL_WALK_FORWARD])
    assert v.level == CertificationLevel.RESEARCH_ONLY


def test_unavailable_data_dominates_other_failures():
    """If REJECTED_UNAVAILABLE_DATA is in the reasons, the level is
    REJECTED_UNAVAILABLE_DATA regardless of other failures."""
    v = verdict_from_reasons([
        FailureReason.REJECTED_UNAVAILABLE_DATA,
        FailureReason.FAIL_WALK_FORWARD,
        FailureReason.FAIL_LOW_PASS_PROBABILITY,
    ])
    assert v.level == CertificationLevel.REJECTED_UNAVAILABLE_DATA


def test_lookahead_caps_at_rejected_broken():
    v = verdict_from_reasons([FailureReason.FAIL_LOOKAHEAD])
    assert v.level == CertificationLevel.REJECTED_BROKEN


def test_prop_specific_failures_cap_at_prop_candidate():
    for r in (FailureReason.FAIL_DAILY_LOSS_LIMIT,
              FailureReason.FAIL_TRAILING_DRAWDOWN,
              FailureReason.FAIL_LOW_PASS_PROBABILITY,
              FailureReason.FAIL_HIGH_BLOWUP_PROBABILITY,
              FailureReason.FAIL_POOR_PAYOUT_SURVIVAL,
              FailureReason.FAIL_ACCOUNT_UNVERIFIED):
        v = verdict_from_reasons([r])
        assert v.level == CertificationLevel.PROP_CANDIDATE, (r, v.level)


def test_promote_demote_clamp_at_extremes():
    assert demote(CertificationLevel.REJECTED_UNAVAILABLE_DATA) == \
        CertificationLevel.REJECTED_UNAVAILABLE_DATA
    assert promote(CertificationLevel.CERTIFIED) == CertificationLevel.CERTIFIED
    # promote climbs by one step until certified, skipping retired
    assert promote(CertificationLevel.WATCHLIST) == CertificationLevel.CANDIDATE
    assert promote(CertificationLevel.PROP_CANDIDATE) == CertificationLevel.CERTIFIED


def test_verdict_round_trips_through_json():
    v = verdict_from_reasons(
        [FailureReason.FAIL_WALK_FORWARD, FailureReason.FAIL_PROFIT_FACTOR],
        detail={"wf_folds": 3, "profit_factor": 0.96},
    )
    payload = v.to_json()
    reborn = CertificationVerdict.from_json(payload)
    assert reborn.level == v.level
    assert reborn.failure_reasons == v.failure_reasons
    assert reborn.detail == v.detail


def test_best_possible_cap_can_lower_ceiling():
    """Caller can request best_possible=PROP_CANDIDATE; if no failures
    then the level is PROP_CANDIDATE (not CERTIFIED)."""
    v = verdict_from_reasons(
        [], best_possible=CertificationLevel.PROP_CANDIDATE)
    assert v.level == CertificationLevel.PROP_CANDIDATE


if __name__ == "__main__":
    fns = [
        test_levels_unique_and_ordered_worst_to_best,
        test_failure_reason_caps_are_consistent,
        test_no_reasons_means_best_possible,
        test_walk_forward_fail_caps_at_research_only,
        test_unavailable_data_dominates_other_failures,
        test_lookahead_caps_at_rejected_broken,
        test_prop_specific_failures_cap_at_prop_candidate,
        test_promote_demote_clamp_at_extremes,
        test_verdict_round_trips_through_json,
        test_best_possible_cap_can_lower_ceiling,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    if failures:
        raise SystemExit(1)
