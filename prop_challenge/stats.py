"""Confidence-interval utilities for prop-challenge outcomes.

Wilson score interval is preferred for binomial proportions over the
naive Wald interval because it stays inside [0,1] for small samples
and small/large probabilities (which is exactly the regime for prop
challenge pass / blowup probabilities).

Bootstrap CI is used when we already have a parametric resampling
distribution and want an empirical CI rather than an analytic one
(e.g. days-to-pass median).
"""
from __future__ import annotations

import math

import numpy as np


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (point_estimate, lower, upper). For n=0 returns (0, 0, 1)
    so callers can check `lower > threshold` without special-casing.
    """
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = successes / n
    # two-sided z; for alpha=0.05 -> ~1.96
    z = abs(_inv_phi(1.0 - alpha / 2.0))
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    lower = (centre - half) / denom
    upper = (centre + half) / denom
    return p, max(0.0, lower), min(1.0, upper)


def _inv_phi(p: float) -> float:
    """Inverse of the standard-normal CDF (Acklam's algorithm).

    Vendored as ~30 lines so we don't take a scipy dependency just for
    one quantile. Accurate to ~1e-9 in [0,1]."""
    # Coefficients from Peter Acklam, 2003.
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def bootstrap_median_ci(values: np.ndarray, n_runs: int = 2000,
                         alpha: float = 0.05,
                         seed: int = 17) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the median of `values`."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.empty(n_runs, dtype=float)
    n = len(values)
    for i in range(n_runs):
        idx = rng.integers(0, n, size=n)
        medians[i] = float(np.median(values[idx]))
    lo = float(np.percentile(medians, 100.0 * alpha / 2.0))
    hi = float(np.percentile(medians, 100.0 * (1.0 - alpha / 2.0)))
    return float(np.median(values)), lo, hi
