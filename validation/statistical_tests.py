"""Statistical significance probes for a strategy:

  1. Shuffled-outcome test:
     Permute the per-trade returns N times, record the sharpe of each
     shuffled series, and compute p_value = P(shuffled_sharpe >= real).

  2. Random-strategy baseline:
     Generate N random direction sequences with the SAME trade
     frequency and apply them on the actual price path. Compute the
     95th percentile sharpe of the random distribution; the strategy
     must exceed it.

  3. False discovery rate (FDR / Benjamini-Hochberg):
     Apply across the full leaderboard p-values to control the
     expected proportion of false discoveries.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.constants import H4_BARS_PER_YEAR
from core.types import Trade


def _annualised_sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    sd = float(rets.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(rets.mean() / sd) * math.sqrt(H4_BARS_PER_YEAR)


def shuffled_outcome_test(trades: list[Trade], n_perm: int = 1000, seed: int = 7) -> dict:
    """How likely is the realised Sharpe under random trade ordering?

    We shuffle the trade returns; this preserves the marginal return
    distribution but destroys any time-structure (e.g. autocorrelation
    or regime grouping). If the strategy's edge depends on hitting the
    right bars (rather than the marginal distribution), this test
    catches it.
    """
    if not trades:
        return {"p_value": 1.0, "passes": False, "n_perm": 0}
    rng = np.random.default_rng(seed)
    rets = np.array([t.ret for t in trades], dtype=float)
    real = _annualised_sharpe(rets)
    sims = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(rets)  # in-place permutation
        sims[i] = _annualised_sharpe(rets)
    # Two-sided would be |sim| >= |real|; we want the upper tail.
    p_upper = float((sims >= real).mean())
    return {
        "p_value": round(p_upper, 4),
        "passes": p_upper < 0.05,
        "n_perm": int(n_perm),
        "real_sharpe": round(real, 3),
        "shuffled_p95_sharpe": round(float(np.percentile(sims, 95)), 3),
    }


def random_baseline_test(real_trades: list[Trade], h4: pd.DataFrame,
                         n_runs: int = 200, seed: int = 11) -> dict:
    """Generate N random direction sequences with the same trade frequency
    on the same price path, compute their Sharpes, and measure where
    the real Sharpe ranks.
    """
    if not real_trades:
        return {"p_value": 1.0, "passes": False}
    n = len(real_trades)
    rng = np.random.default_rng(seed)
    h4 = h4.sort_values("time").reset_index(drop=True)
    n_h4 = len(h4)
    if n_h4 < n + 10:
        return {"p_value": 1.0, "passes": False, "note": "not enough bars"}

    bar_ret = (h4["close"].values - h4["open"].values) / h4["open"].values

    real_sharpe = _annualised_sharpe(np.array([t.ret for t in real_trades]))
    sims = np.empty(n_runs)
    for r in range(n_runs):
        idxs = rng.choice(n_h4, size=n, replace=False)
        dirs = rng.choice([-1, 1], size=n)
        rets = dirs * bar_ret[idxs]
        sims[r] = _annualised_sharpe(rets)
    p = float((sims >= real_sharpe).mean())
    p95 = float(np.percentile(sims, 95))
    return {
        "p_value": round(p, 4),
        "passes": (p < 0.05) and (real_sharpe > p95),
        "real_sharpe": round(real_sharpe, 3),
        "random_p95_sharpe": round(p95, 3),
        "n_runs": int(n_runs),
    }


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Return a same-length list of "is significant under FDR <= q" booleans."""
    if not p_values:
        return []
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    sig_ranked = ranked <= thresholds
    if not sig_ranked.any():
        keep = np.zeros(n, dtype=bool)
    else:
        # largest k where p_(k) <= (k/n) q
        k = int(np.where(sig_ranked)[0].max())
        keep_ranked = np.arange(n) <= k
        keep = np.zeros(n, dtype=bool)
        keep[order[:k + 1]] = True
    return keep.tolist()
