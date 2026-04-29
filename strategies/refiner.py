"""Refiner — map failure reasons -> next-mutation suggestions.

The Batch H orchestrator calls `propose_mutations(candidate)` for
each candidate that didn't certify. The refiner reads the
candidate's `failure_reasons` and emits a small list of mutation
suggestions, each as either a callable (returns a fresh
PropCandidate) or a structured plan dict.

This is intentionally lightweight. The user's brief lays out the
mapping verbatim:

  daily loss breach        -> stricter daily lockout
  too many trades          -> max_trades_per_day cap
  low pass / low blowup    -> slightly higher risk
  high blowup              -> lower risk OR stricter lockout
  one-session edge         -> isolate that session
  too few trades           -> widen entry / add related setup
  failed random baseline   -> reject / redesign setup
  failed spread stress     -> better entry / larger target

We translate these into concrete PropCandidate variants by reusing
the Batch G labs.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.candidate import DailyRulesBlock, PropCandidate, RiskBlock
from core.certification import FailureReason

from strategies import daily_rule_lab, entry_lab, risk_sweep


@dataclass
class MutationSuggestion:
    """One concrete next-step variant + the reason it was suggested."""
    candidate: PropCandidate
    rationale: str


def _clone_with(base: PropCandidate, *,
                 risk: RiskBlock | None = None,
                 daily: DailyRulesBlock | None = None,
                 id_suffix: str = "") -> PropCandidate:
    return PropCandidate(
        id=f"{base.id}__refined_{id_suffix}" if id_suffix else f"{base.id}__refined",
        family=base.family,
        symbol=base.symbol,
        bias_timeframe=base.bias_timeframe,
        setup_timeframe=base.setup_timeframe,
        entry_timeframe=base.entry_timeframe,
        signal=dict(base.signal),
        filters=[dict(f) for f in base.filters],
        entry=dict(base.entry),
        stop=dict(base.stop),
        exit=dict(base.exit),
        risk=risk or base.risk,
        daily_rules=daily or base.daily_rules,
        account=base.account,
        cost_model=base.cost_model,
        cost_bps=base.cost_bps,
        seed=base.seed,
        notes=base.notes,
        provenance=dict(base.provenance, lab="refiner"),
    )


def _stricter_daily(base: PropCandidate) -> list[MutationSuggestion]:
    """Suggest tighter daily lockouts."""
    out = []
    cur = base.daily_rules
    # Cap max trades at 1 (stricter than current)
    out.append(MutationSuggestion(
        _clone_with(base,
                    daily=DailyRulesBlock(name="max1",
                                            max_trades_per_day=1),
                    id_suffix="max1"),
        rationale="cap to 1 trade/day to limit daily loss exposure"))
    # Stop after 1 loss
    out.append(MutationSuggestion(
        _clone_with(base,
                    daily=DailyRulesBlock(name="stop_l1",
                                            stop_after_n_losses=1),
                    id_suffix="stop_l1"),
        rationale="stop trading after the first loss of the day"))
    # Cooldown after loss (60 min)
    out.append(MutationSuggestion(
        _clone_with(base,
                    daily=DailyRulesBlock(name="cd60",
                                            cooldown_minutes_after_loss=60),
                    id_suffix="cd60"),
        rationale="60-minute cooldown after a loss to break the chase"))
    return out


def _lower_risk(base: PropCandidate) -> list[MutationSuggestion]:
    out = []
    out.append(MutationSuggestion(
        _clone_with(base,
                    risk=RiskBlock(name="micro_1", contracts_base=1,
                                    contracts_max=base.risk.contracts_max),
                    id_suffix="risk_micro_1"),
        rationale="halve contracts to reduce blow-up probability"))
    out.append(MutationSuggestion(
        _clone_with(base,
                    risk=RiskBlock(name="dollar_risk_50",
                                    dollar_risk_per_trade=50.0,
                                    contracts_max=base.risk.contracts_max),
                    id_suffix="risk_dr50"),
        rationale="$50 fixed dollar-risk per trade"))
    return out


def _higher_risk(base: PropCandidate) -> list[MutationSuggestion]:
    """For high-pass, low-blowup strategies that under-perform on
    point estimates: bump risk a notch."""
    out = []
    cur_max = base.risk.contracts_max
    out.append(MutationSuggestion(
        _clone_with(base,
                    risk=RiskBlock(name="dollar_risk_100",
                                    dollar_risk_per_trade=100.0,
                                    contracts_max=cur_max),
                    id_suffix="risk_dr100"),
        rationale="raise dollar-risk to $100 to capture more of the edge"))
    return out


def _isolate_session(base: PropCandidate) -> list[MutationSuggestion]:
    out = []
    for session in ("ny", "london", "ny_open", "london_open"):
        out.append(MutationSuggestion(
            _clone_with(base,
                        daily=DailyRulesBlock(name=f"{session}_only",
                                                session_only=session),
                        id_suffix=f"session_{session}"),
            rationale=f"isolate trading to the {session} session"))
    return out


def _widen_entry(base: PropCandidate) -> list[MutationSuggestion]:
    """Too few trades -> try alternate entry models that fire more often."""
    return [MutationSuggestion(c,
                                  rationale=f"widen entry to {c.entry.get('type')}")
            for c in entry_lab.variants(base, entries=(
                {"type": "touch_entry"},
                {"type": "minor_structure_break"},
                {"type": "delayed_entry_1"},
            ))]


def propose_mutations(candidate: PropCandidate
                       ) -> list[MutationSuggestion]:
    """Read the candidate's failure_reasons and emit a small batch
    of mutation suggestions. Empty list if no actionable mapping
    applies (e.g. REJECTED_UNAVAILABLE_DATA — no mutation can fix
    that, the user has to add real volume to the dataset)."""
    out: list[MutationSuggestion] = []
    reasons = set(candidate.failure_reasons)

    if FailureReason.REJECTED_UNAVAILABLE_DATA in reasons:
        return []      # not actionable; data is missing

    if (FailureReason.FAIL_DAILY_LOSS_LIMIT in reasons or
            FailureReason.FAIL_TRAILING_DRAWDOWN in reasons or
            FailureReason.FAIL_HIGH_BLOWUP_PROBABILITY in reasons):
        out.extend(_stricter_daily(candidate))
        out.extend(_lower_risk(candidate))

    if FailureReason.FAIL_TOO_MANY_TRADES in reasons:
        out.extend(_stricter_daily(candidate))

    if FailureReason.FAIL_TOO_FEW_TRADES in reasons:
        out.extend(_widen_entry(candidate))

    if FailureReason.FAIL_LOW_PASS_PROBABILITY in reasons \
            and FailureReason.FAIL_HIGH_BLOWUP_PROBABILITY not in reasons:
        out.extend(_higher_risk(candidate))

    if FailureReason.FAIL_ONE_SESSION_DEPENDENCY in reasons:
        out.extend(_isolate_session(candidate))

    if FailureReason.FAIL_RANDOM_BASELINE in reasons \
            or FailureReason.FAIL_LABEL_PERMUTATION in reasons:
        # No edge attributable to direction; refining doesn't help.
        # Suggest re-design / family change rather than parameter tweak.
        return [MutationSuggestion(
            candidate,
            rationale=("REJECT_AND_REDESIGN: failed random / label "
                        "permutation gate; the directional edge is "
                        "indistinguishable from random. Suggest "
                        "moving to a different strategy family.")
        )]

    return out


def summarise(suggestions: list[MutationSuggestion]) -> dict:
    """Plain-data summary for the report. Used by Batch H/I."""
    return {
        "n_suggestions": len(suggestions),
        "suggestions": [{"candidate_id": s.candidate.id,
                          "rationale": s.rationale}
                         for s in suggestions],
    }
