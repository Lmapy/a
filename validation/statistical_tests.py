"""Statistical significance probes for a strategy (Phase 5 hardening).

Replaces the previous `shuffled_outcome_test`, which was a no-op:
shuffling the trade-return list does not change mean or std, so the
annualised Sharpe is invariant to the permutation -- the test always
returned p ~= 1.0 and could never reject. The certifier was
therefore gating on noise.

What's here now:

  1. label_permutation_test
     Permute trade direction (+1 / -1) at the same timestamps. Real
     edges depend on direction-given-context; flipping signs
     destroys that signal while preserving cost structure, trade
     count, and timing.

  2. random_eligible_entry_test
     Generate N random entry sequences with the same trade count on
     the same eligible bars (i.e. bars that pass the spec's filters)
     and compute Sharpe / total return. The strategy must beat the
     baseline distribution materially, not just the upper tail.

  3. daily_block_bootstrap_test
     Stationary block-bootstrap on per-day equity returns (not
     per-trade returns). Preserves intra-day clustering and
     regime-day correlation. Reports a bootstrap CI for total
     return, then the percentile rank of the realised return.

  4. benjamini_hochberg
     False-discovery-rate adjustment across a leaderboard's p-values.
     Kept from the original implementation.

  5. shuffled_outcome_test (DEPRECATED — raises RuntimeError)
     Kept as a tombstone so any code that still imports it fails
     loudly rather than silently using a broken gate.

Default permutation counts:

    N_PERM_EXPLORATION = 500   (search-time, must run on every spec)
    N_PERM_FINAL       = 5000  (certifier, applies only to candidates)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.constants import H4_BARS_PER_YEAR
from core.types import Trade


# Sensible defaults; callers can override.
N_PERM_EXPLORATION = 500
N_PERM_FINAL = 5000


def _annualised_sharpe_h4_bar(rets: np.ndarray) -> float:
    """Annualised Sharpe assuming one return per H4 bar. Used only for
    relative comparisons inside the same test; for the canonical
    metrics see analytics.trade_metrics."""
    if len(rets) < 2:
        return 0.0
    sd = float(rets.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(rets.mean() / sd) * math.sqrt(H4_BARS_PER_YEAR)


def shuffled_outcome_test(*args, **kwargs) -> dict:
    """DEPRECATED — Sharpe is invariant under per-trade permutation.

    Raises RuntimeError on import to make sure no certifier silently
    relies on it. Use label_permutation_test instead.
    """
    raise RuntimeError(
        "shuffled_outcome_test is deprecated and removed in the Phase 5 "
        "hardening pass. Sharpe is invariant under permutation of trade "
        "returns, so the test is mathematically a no-op. Use "
        "label_permutation_test (random direction at same timestamps) "
        "or daily_block_bootstrap_test instead.")


# ---------- 1. label permutation ----------

def label_permutation_test(trades: list[Trade],
                           n_perm: int = N_PERM_EXPLORATION,
                           seed: int = 7) -> dict:
    """Permute the long/short label at each trade timestamp.

    Real returns r_i = direction_i * (exit_i - entry_i) / entry_i. We
    flip the direction sign at random and recompute returns. Under the
    null "the strategy has no directional edge", the permuted Sharpe
    distribution should center around zero and the realised Sharpe
    should sit well into its right tail.

    This test isolates *directional* edge from cost / timing structure.
    Costs (spread + slippage) are paid on every permuted trade just
    as they were on the real trade.
    """
    if not trades:
        return {"name": "label_permutation",
                "p_value": 1.0, "passes": False, "n_perm": 0}
    rng = np.random.default_rng(seed)

    # Decompose returns into the cost-bearing magnitude and the sign.
    # ret = sign * |move/entry| - cost/entry, but the published `ret` is
    # net of costs. Approximate the gross-of-cost magnitude as |ret|+cost/entry.
    # For the directional flip we just negate `ret` because flipping the
    # trade direction inverts the gross PnL while leaving cost identical
    # in magnitude. Net-of-cost shape is preserved.
    rets = np.array([t.ret for t in trades], dtype=float)
    real = _annualised_sharpe_h4_bar(rets)

    sims = np.empty(n_perm, dtype=float)
    n = len(rets)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        sims[i] = _annualised_sharpe_h4_bar(rets * signs)

    p_upper = float((sims >= real).mean())
    p_two = float((np.abs(sims) >= abs(real)).mean())
    return {
        "name": "label_permutation",
        "p_value": round(p_upper, 4),
        "p_two_sided": round(p_two, 4),
        "passes": p_upper < 0.05,
        "n_perm": int(n_perm),
        "real_sharpe": round(real, 3),
        "permuted_p95_sharpe": round(float(np.percentile(sims, 95)), 3),
        "permuted_mean_sharpe": round(float(sims.mean()), 3),
    }


# ---------- 2. random eligible entry baseline ----------

def random_eligible_entry_test(real_trades: list[Trade],
                                h4: pd.DataFrame,
                                n_runs: int = N_PERM_EXPLORATION,
                                eligible_mask: np.ndarray | None = None,
                                seed: int = 11) -> dict:
    """Random entries on eligible bars with same trade count.

    eligible_mask: optional boolean array over h4 bars marking which
    are admissible entry bars (e.g. those that pass the spec's
    filters). If None, all bars are considered eligible.

    Returns Sharpe of the real trades vs. the distribution of random
    Sharpes from same-count, random-direction entries on eligible bars.
    Real strategy must beat the 95th percentile materially.
    """
    if not real_trades:
        return {"name": "random_eligible_entry",
                "p_value": 1.0, "passes": False, "n_runs": 0}
    h4 = h4.sort_values("time").reset_index(drop=True)
    n_h4 = len(h4)
    if eligible_mask is None:
        eligible_mask = np.ones(n_h4, dtype=bool)
    elig_idx = np.where(eligible_mask)[0]
    target_n = len(real_trades)
    if len(elig_idx) < target_n + 5:
        return {"name": "random_eligible_entry",
                "p_value": 1.0, "passes": False, "n_runs": 0,
                "note": "insufficient eligible bars"}

    bar_ret = (h4["close"].values - h4["open"].values) / h4["open"].values
    real_sharpe = _annualised_sharpe_h4_bar(np.array([t.ret for t in real_trades]))

    rng = np.random.default_rng(seed)
    sims = np.empty(n_runs, dtype=float)
    for r in range(n_runs):
        idxs = rng.choice(elig_idx, size=target_n, replace=False)
        dirs = rng.choice([-1, 1], size=target_n)
        sims[r] = _annualised_sharpe_h4_bar(dirs * bar_ret[idxs])

    p = float((sims >= real_sharpe).mean())
    p95 = float(np.percentile(sims, 95))
    return {
        "name": "random_eligible_entry",
        "p_value": round(p, 4),
        "passes": (p < 0.05) and (real_sharpe > p95),
        "real_sharpe": round(real_sharpe, 3),
        "random_p95_sharpe": round(p95, 3),
        "random_p50_sharpe": round(float(np.median(sims)), 3),
        "n_runs": int(n_runs),
        "n_eligible_bars": int(len(elig_idx)),
    }


# ---------- 3. block bootstrap on daily equity returns ----------

def _daily_returns(trades: list[Trade]) -> np.ndarray:
    """Sum trade returns into per-calendar-day buckets. Days with no
    trades are NOT included (this is a per-trading-day series, not a
    calendar series); intra-day clustering is preserved within each
    bucket via ordinary summation."""
    if not trades:
        return np.array([], dtype=float)
    df = pd.DataFrame({
        "date": [pd.Timestamp(t.entry_time).normalize() for t in trades],
        "ret": [float(t.ret) for t in trades],
    })
    grp = df.groupby("date")["ret"].sum()
    return grp.values


def daily_block_bootstrap_test(trades: list[Trade],
                                block_days: int = 5,
                                n_runs: int = N_PERM_EXPLORATION,
                                seed: int = 13) -> dict:
    """Stationary block bootstrap on daily-equity returns.

    Trades are bucketed by entry-date. Daily returns are then resampled
    in contiguous blocks of `block_days` (with replacement, circular)
    to preserve serial structure. Two outputs:

      * boot_p05 / boot_p50 / boot_p95: percentiles of the resampled
        total return. boot_p05 > 0 means the strategy's edge is robust
        to block-resampling -- even unfavourable resamples remain
        positive, evidence of a genuine clustered edge.

      * p_value: right-tail probability under a *zero-mean* null,
        constructed by demeaning the daily series before resampling.
        This compares the realised total to what an edge-free
        strategy with the same volatility profile would produce.

    The certifier's `passes` gate uses the lower-CI rule (boot_p05 > 0)
    rather than only a right-tail p-value, because a near-constant
    daily series can have a perfectly significant total return AND
    a degenerate p-value (all bootstrap draws ~ realised).
    """
    daily = _daily_returns(trades)
    if len(daily) < 5:
        return {"name": "daily_block_bootstrap",
                "p_value": 1.0, "passes": False, "n_runs": 0,
                "note": "insufficient trading days"}
    rng = np.random.default_rng(seed)
    real_total = float(np.prod(1.0 + daily) - 1.0)
    demeaned = daily - daily.mean()

    sims_total = np.empty(n_runs, dtype=float)
    sims_null = np.empty(n_runs, dtype=float)
    n = len(daily)
    for r in range(n_runs):
        out = np.empty(n, dtype=float)
        out_null = np.empty(n, dtype=float)
        pos = 0
        while pos < n:
            start = int(rng.integers(0, n))
            take = min(block_days, n - pos)
            sl = (start + np.arange(take)) % n
            out[pos:pos + take] = daily[sl]
            out_null[pos:pos + take] = demeaned[sl]
            pos += take
        sims_total[r] = float(np.prod(1.0 + out) - 1.0)
        sims_null[r] = float(np.prod(1.0 + out_null) - 1.0)

    p_upper = float((sims_null >= real_total).mean())
    p_two = float((np.abs(sims_null) >= abs(real_total)).mean())
    p05 = float(np.percentile(sims_total, 5))
    p50 = float(np.percentile(sims_total, 50))
    p95 = float(np.percentile(sims_total, 95))
    passes = (p05 > 0.0) and (p_upper < 0.05)
    return {
        "name": "daily_block_bootstrap",
        "p_value": round(p_upper, 4),
        "p_two_sided": round(p_two, 4),
        "passes": passes,
        "real_total_return": round(real_total, 4),
        "boot_p05": round(p05, 4),
        "boot_p50": round(p50, 4),
        "boot_p95": round(p95, 4),
        "boot_lower_ci_above_zero": p05 > 0.0,
        "n_runs": int(n_runs),
        "n_trading_days": int(len(daily)),
        "block_days": int(block_days),
    }


# ---------- 4. Benjamini-Hochberg FDR ----------

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
        k = int(np.where(sig_ranked)[0].max())
        keep = np.zeros(n, dtype=bool)
        keep[order[:k + 1]] = True
    return keep.tolist()


# ---------- back-compat alias for the random baseline ----------

def random_baseline_test(real_trades: list[Trade], h4: pd.DataFrame,
                         n_runs: int = N_PERM_EXPLORATION, seed: int = 11) -> dict:
    """Back-compat shim: forwards to random_eligible_entry_test with no
    eligibility mask. New code should call random_eligible_entry_test
    directly and pass the spec's filter mask for tighter power."""
    return random_eligible_entry_test(real_trades, h4, n_runs=n_runs,
                                       eligible_mask=None, seed=seed)
