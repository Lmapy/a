"""Prop passing score formula.

The user's brief defines the composite ranking metric:

    Prop Passing Score =
        + pass_probability                          * 35
        + payout_survival_probability               * 20
        + drawdown_safety_score                     * 20
        + consistency_score                         * 10
        + median_days_score                         * 10
        + simplicity_score                          * 5
        - blowup_probability                        * 35
        - daily_loss_breach_probability             * 20
        - trailing_drawdown_breach_probability      * 20
        - overfit_penalty                           * 25

All input values are in [0, 1]. The output is unbounded (in
practice it lives in roughly [-100, +100]; positive = good).

We intentionally do NOT normalise to [0, 1] because the rank-only
property is what matters; preserving the raw additive form makes
it easy to see which factor dominated for a given candidate.

Helper definitions
------------------
  drawdown_safety_score(max_drawdown) = clamp(1 + max_drawdown / 0.20, 0, 1)
      max_drawdown is negative; 0% DD -> 1.0, -20% -> 0.0
  consistency_score(yearly_positive, yearly_total)
      = yearly_positive / max(yearly_total, 1)
  median_days_score(median_days, target_days=30)
      = clamp((target_days - median_days) / target_days, 0, 1)
      so median 0 days = 1.0, median 30 days = 0, > 30 = 0.
  simplicity_score(n_filters, target=2)
      = clamp(1 - max(0, n_filters - target) / 4, 0, 1)
      few filters -> 1.0, more than 6 filters -> 0.
  overfit_penalty(label_perm_p, random_p)
      = max(0, max(label_perm_p, random_p) - 0.05) / 0.45
      penalises p-values above 0.05.
"""
from __future__ import annotations


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def drawdown_safety_score(max_drawdown: float, *, dd_floor: float = 0.20) -> float:
    """max_drawdown is negative or zero. dd_floor is the
    "completely unacceptable" magnitude (default 20%)."""
    if max_drawdown >= 0:
        return 1.0
    return _clamp(1.0 + max_drawdown / dd_floor)


def consistency_score(yearly_positive: int, yearly_total: int) -> float:
    if yearly_total <= 0:
        return 0.0
    return _clamp(yearly_positive / yearly_total)


def median_days_score(median_days_to_pass: float | None,
                       target_days: int = 30) -> float:
    if median_days_to_pass is None:
        return 0.0
    return _clamp((target_days - median_days_to_pass) / target_days)


def simplicity_score(n_filters: int, target: int = 2) -> float:
    return _clamp(1.0 - max(0, n_filters - target) / 4.0)


def overfit_penalty(label_perm_p: float | None,
                     random_p: float | None,
                     block_boot_p: float | None = None) -> float:
    """Larger p-values -> bigger penalty. Returns [0, 1]."""
    ps = [p for p in (label_perm_p, random_p, block_boot_p)
          if p is not None]
    if not ps:
        return 1.0
    p_max = max(ps)
    return _clamp(max(0.0, p_max - 0.05) / 0.45)


def prop_passing_score(*,
                        pass_probability: float,
                        blowup_probability: float,
                        payout_survival_probability: float,
                        daily_loss_breach_probability: float,
                        trailing_drawdown_breach_probability: float,
                        max_drawdown: float,
                        yearly_positive: int,
                        yearly_total: int,
                        median_days_to_pass: float | None,
                        n_filters: int,
                        label_perm_p: float | None,
                        random_p: float | None,
                        block_boot_p: float | None = None) -> dict:
    """Compute the prop passing score and the constituent components.

    Returns a dict with `score` and the per-axis sub-scores, so the
    leaderboard can show why a candidate ranks where it does."""
    dd_safe = drawdown_safety_score(max_drawdown)
    cons = consistency_score(yearly_positive, yearly_total)
    days = median_days_score(median_days_to_pass)
    simp = simplicity_score(n_filters)
    overfit = overfit_penalty(label_perm_p, random_p, block_boot_p)

    score = (
        pass_probability                       * 35
        + payout_survival_probability           * 20
        + dd_safe                               * 20
        + cons                                  * 10
        + days                                  * 10
        + simp                                  * 5
        - blowup_probability                    * 35
        - daily_loss_breach_probability         * 20
        - trailing_drawdown_breach_probability  * 20
        - overfit                               * 25
    )
    return {
        "score": round(score, 3),
        "components": {
            "pass_probability": round(pass_probability, 4),
            "payout_survival_probability": round(payout_survival_probability, 4),
            "drawdown_safety_score": round(dd_safe, 4),
            "consistency_score": round(cons, 4),
            "median_days_score": round(days, 4),
            "simplicity_score": round(simp, 4),
            "blowup_probability": round(blowup_probability, 4),
            "daily_loss_breach_probability": round(daily_loss_breach_probability, 4),
            "trailing_drawdown_breach_probability": round(trailing_drawdown_breach_probability, 4),
            "overfit_penalty": round(overfit, 4),
        },
    }
