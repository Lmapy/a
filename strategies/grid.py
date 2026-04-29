"""Tier-1 grid composer.

A "tier-1" grid is the FAST first-pass set of candidates the
orchestrator runs through OHLC backtest + walk-forward. It must be
controlled in size — running every family x every entry x every
risk x every daily rule x every account is ~18,000 combinations
and would never finish.

Tiered runner contract
----------------------
The orchestrator (Batch H, `scripts/run_prop_passing.py`) flows:

  1. tier_1_grid()       families x small risk/daily/entry default
                          (~150-300 candidates)
  2. backtest + filter   reject capability failures + obvious losers
  3. tier_2_sweep()      survivors x wider risk/daily/entry sweep
  4. prop sim            survivors x verified accounts
  5. judge + leaderboard

Tier 1 caps roughly at:

  10 families x ~5 base variants per family   = ~50 base candidates
  x 1 default risk model                       = 50
  x 1 default daily rule                       = 50

Tier 2 adds (only on survivors):

  + 7 risk presets x 3 contracts caps           = +21 per survivor
  + 12 daily rule presets                       = +12 per survivor
  + 9 entry models                              = +9 per survivor

Per the user's brief: "Keep the candidate grid controlled."

Helpers
-------
  tier_1_grid(families=None) -> list[PropCandidate]
      All families' base variants. The orchestrator runs these
      against OHLC backtest + walk-forward in parallel.

  apply_capability_filter(candidates) -> (kept, rejected)
      Tag any unavailable-data candidates with REJECTED_UNAVAILABLE_DATA
      and partition the input list. Used at the start of every run.

  tier_2_for_survivors(survivors, *, lab="risk"|"daily"|"entry")
      One survivor in -> N variants out. The orchestrator chooses
      which lab to run per survivor based on the failure mode (e.g.
      "high blowup" -> risk sweep; "daily loss breach" -> daily-rule
      sweep).
"""
from __future__ import annotations

from typing import Iterable, Literal

from core.candidate import PropCandidate
from core.certification import CertificationLevel, FailureReason
from core.feature_capability import classify_candidate

from strategies import daily_rule_lab, entry_lab, families, risk_sweep


def tier_1_grid(*, families_filter: Iterable[str] | None = None
                ) -> list[PropCandidate]:
    """All families' base variants (no entry/risk/daily-rule sweep)."""
    return families.all_candidates(families=families_filter)


def apply_capability_filter(candidates: list[PropCandidate]
                              ) -> tuple[list[PropCandidate], list[PropCandidate]]:
    """Mark candidates whose tokens require unavailable data.

    Returns `(kept, rejected)`. The rejected candidates have their
    `certification_level` set to `REJECTED_UNAVAILABLE_DATA` and
    carry the `REJECTED_UNAVAILABLE_DATA` failure reason; their
    `rejection_detail` lists the offending tokens and rename hints.
    """
    kept: list[PropCandidate] = []
    rejected: list[PropCandidate] = []
    for c in candidates:
        v = classify_candidate(c.to_spec())
        if v.status == "ok":
            kept.append(c)
            continue
        c.certification_level = CertificationLevel.REJECTED_UNAVAILABLE_DATA
        if FailureReason.REJECTED_UNAVAILABLE_DATA not in c.failure_reasons:
            c.failure_reasons.append(FailureReason.REJECTED_UNAVAILABLE_DATA)
        c.rejection_detail.update({
            "unavailable_features": v.unavailable_features,
            "unavailable_tokens": v.unavailable_tokens,
            "rename_hints": v.rename_hints,
        })
        rejected.append(c)
    return kept, rejected


# ---- tier-2 helpers --------------------------------------------------------

LabName = Literal["risk", "daily", "entry"]


def tier_2_for_survivor(candidate: PropCandidate, *,
                          lab: LabName = "risk",
                          contracts_max: int = 5,
                          ) -> list[PropCandidate]:
    """One survivor -> variants on the requested axis only.

    The orchestrator typically runs a survivor against multiple labs
    sequentially: risk first (cheapest, most impactful for prop sim),
    then daily, then entry.
    """
    if lab == "risk":
        return risk_sweep.variants(candidate, contracts_max=contracts_max)
    if lab == "daily":
        return daily_rule_lab.variants(candidate)
    if lab == "entry":
        return entry_lab.variants(candidate)
    raise ValueError(f"unknown lab {lab!r}; must be risk/daily/entry")


def tier_2_full(candidates: list[PropCandidate], *,
                 contracts_max: int = 5) -> list[PropCandidate]:
    """All three labs applied. Use sparingly — output explodes.

    For each candidate produces:
        risk variants (~7) + daily variants (~12) + entry variants (~9)
    -> 28 per survivor. With 50 survivors that's 1,400 candidates.
    """
    out: list[PropCandidate] = []
    for c in candidates:
        out.extend(risk_sweep.variants(c, contracts_max=contracts_max))
        out.extend(daily_rule_lab.variants(c))
        out.extend(entry_lab.variants(c))
    return out


# ---- diagnostics -----------------------------------------------------------

def grid_summary(candidates: list[PropCandidate]) -> dict:
    """A plain-data summary of a candidate set. Used by the runner's
    progress reporter."""
    by_family: dict[str, int] = {}
    by_cert: dict[str, int] = {}
    for c in candidates:
        by_family[c.family] = by_family.get(c.family, 0) + 1
        lvl = c.certification_level.value
        by_cert[lvl] = by_cert.get(lvl, 0) + 1
    return {
        "total": len(candidates),
        "by_family": by_family,
        "by_certification_level": by_cert,
    }
