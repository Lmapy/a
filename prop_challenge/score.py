"""prop_score combines challenge + payout outcomes into one number,
then `passes_cert` enforces the strict prop-firm certification gate
on the LOWER 95% CI bound (not the point estimate).

Phase 7 hardening: the previous certifier compared point estimates to
thresholds. With Monte Carlo at 500-5000 runs, the sampling noise on
the point estimate is non-trivial; gating on a noisy point estimate
inflates type-I (false-pass) rate. We now require that the lower
Wilson CI bound clears the threshold for pass / payout, and the upper
CI bound stays under the threshold for blowup. Median-days-to-pass
gates on the upper CI bound of the bootstrap median.
"""
from __future__ import annotations

from prop_challenge.challenge import ChallengeResult
from prop_challenge.payout import PayoutResult


def prop_score(challenge: ChallengeResult, payout: PayoutResult) -> float:
    s = challenge.pass_probability
    s -= 0.5 * challenge.blowup_probability
    s -= 0.3 * challenge.consistency_breach_rate * challenge.pass_probability
    rule_share = sum(challenge.breach_reason_histogram.values()) / max(challenge.n_runs, 1)
    s -= 0.2 * rule_share
    if challenge.median_days_to_pass is not None and challenge.median_days_to_pass < 5:
        s -= 0.1
    s += 0.5 * payout.first_payout_probability
    return round(s, 4)


def passes_cert(challenge: ChallengeResult, payout: PayoutResult,
                *,
                min_pass: float = 0.35,
                min_payout: float = 0.20,
                max_blowup: float = 0.15,
                max_consistency: float = 0.15,
                max_median_days: int = 30,
                use_ci_bounds: bool = True) -> tuple[bool, list[str]]:
    """If `use_ci_bounds` (default), gate on:
        pass:    lower CI of pass_probability        >= min_pass
        payout:  lower CI of first_payout_probability>= min_payout
        blowup:  upper CI of blowup_probability      <= max_blowup
        median_days: upper CI of bootstrap median    <= max_median_days
    Else falls back to point-estimate gating (legacy)."""
    fails: list[str] = []

    if use_ci_bounds:
        pp_lo = challenge.pass_probability_ci[0]
        if pp_lo < min_pass:
            fails.append(
                f"pass_prob lower CI {pp_lo:.2%} < {min_pass:.0%} "
                f"(point {challenge.pass_probability:.2%})")
        po_lo = payout.first_payout_probability_ci[0]
        if po_lo < min_payout:
            fails.append(
                f"payout_prob lower CI {po_lo:.2%} < {min_payout:.0%} "
                f"(point {payout.first_payout_probability:.2%})")
        bp_hi = challenge.blowup_probability_ci[1]
        if bp_hi > max_blowup:
            fails.append(
                f"blowup_prob upper CI {bp_hi:.2%} > {max_blowup:.0%} "
                f"(point {challenge.blowup_probability:.2%})")
        if challenge.consistency_breach_rate > max_consistency:
            fails.append(
                f"consistency_breach={challenge.consistency_breach_rate:.2%} > {max_consistency:.0%}")
        if challenge.median_days_to_pass is None:
            fails.append("median_days_to_pass is None (no passes)")
        else:
            if challenge.median_days_to_pass_ci is not None:
                upper = challenge.median_days_to_pass_ci[1]
                if upper > max_median_days:
                    fails.append(
                        f"median_days upper CI {upper} > {max_median_days} "
                        f"(point {challenge.median_days_to_pass})")
            elif challenge.median_days_to_pass > max_median_days:
                fails.append(f"median_days={challenge.median_days_to_pass} > {max_median_days}")
    else:
        # legacy point-estimate gating
        if challenge.pass_probability < min_pass:
            fails.append(f"pass_prob={challenge.pass_probability:.2%} < {min_pass:.0%}")
        if payout.first_payout_probability < min_payout:
            fails.append(f"payout_prob={payout.first_payout_probability:.2%} < {min_payout:.0%}")
        if challenge.blowup_probability > max_blowup:
            fails.append(f"blowup_prob={challenge.blowup_probability:.2%} > {max_blowup:.0%}")
        if challenge.consistency_breach_rate > max_consistency:
            fails.append(f"consistency_breach={challenge.consistency_breach_rate:.2%} > {max_consistency:.0%}")
        if challenge.median_days_to_pass is None:
            fails.append("median_days_to_pass is None (no passes)")
        elif challenge.median_days_to_pass > max_median_days:
            fails.append(f"median_days={challenge.median_days_to_pass} > {max_median_days}")

    return (not fails, fails)
