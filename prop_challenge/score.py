"""prop_score combines challenge + payout outcomes into one number.

  prop_score = pass_probability
             - blowup_penalty           (0.5 * blowup_probability)
             - consistency_penalty      (0.3 * consistency_breach_rate * pass_probability)
             - rule_violation_penalty   (sum of breach reasons / n_runs * 0.2)
             - overtrading_penalty      (0.1 if median_days_to_pass < 5 else 0.0)
             + payout_bonus             (0.5 * first_payout_probability)

Higher is better. Strict cert applies on top:

  pass_probability         >= 0.35
  first_payout_probability >= 0.20
  blowup_probability       <= 0.15
  consistency_breach_rate  <= 0.15
  median_days_to_pass      <= 30
  works on at least 2 distinct account models
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
                *, min_pass: float = 0.35, min_payout: float = 0.20,
                max_blowup: float = 0.15, max_consistency: float = 0.15,
                max_median_days: int = 30) -> tuple[bool, list[str]]:
    fails: list[str] = []
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
