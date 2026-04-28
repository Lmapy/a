"""Prop-firm survivability simulation.

Models 25k / 50k / 150k account presets with daily loss limits and
trailing drawdowns. For each account we Monte Carlo the trade sequence
n_runs times (block-bootstrap) and report:

  - prop_survival_score  (1.0 - blowup_probability)
  - blowup_probability   share of runs that breach a limit
  - end_balance_p10/p50/p90
  - passes_*             whether the median run passes (no blowup)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.constants import POINT_SIZE, PROP_ACCOUNTS
from core.types import Trade


def _block_bootstrap(rets: np.ndarray, block: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap with fixed block length."""
    if len(rets) == 0:
        return rets
    out = np.empty(n)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, len(rets)))
        take = min(block, n - pos)
        out[pos:pos + take] = rets[(start + np.arange(take)) % len(rets)]
        pos += take
    return out


@dataclass
class PropResult:
    account: str
    n_runs: int
    blowup_probability: float
    prop_survival_score: float
    end_balance_p10: float
    end_balance_p50: float
    end_balance_p90: float
    passes: bool


def simulate_account(trades: list[Trade], account_name: str,
                     n_runs: int = 1000, block: int = 5,
                     contract_dollar_pnl_per_unit: float = 1.0,
                     seed: int = 9) -> PropResult:
    """contract_dollar_pnl_per_unit: how many account dollars one unit of
    Trade.pnl (in price units) represents. For XAUUSD futures-equivalent
    sizing on a $1/point contract this is 1.0; tune to match your contract.
    """
    cfg = PROP_ACCOUNTS[account_name]
    if not trades:
        return PropResult(account_name, 0, 0.0, 0.0, cfg["balance"],
                          cfg["balance"], cfg["balance"], False)

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades], dtype=float) * contract_dollar_pnl_per_unit

    # day grouping for daily-loss-limit (use entry date)
    day_idx = pd.Series([pd.Timestamp(t.entry_time).normalize() for t in trades])
    # bootstrapped sequences
    blowups = 0
    end_balances = np.empty(n_runs)
    for r in range(n_runs):
        seq = _block_bootstrap(pnls, block=block, n=len(pnls), rng=rng)
        bal = cfg["balance"]
        peak = bal
        broke = False
        # randomly assign 6-12 trades per day (gold trades 6 H4 bars/day)
        i = 0
        while i < len(seq):
            day_take = int(rng.integers(4, 12))
            day_chunk = seq[i:i + day_take]
            i += day_take
            day_pnl = float(day_chunk.sum())
            if day_pnl < -cfg["daily_loss_limit"]:
                broke = True
                break
            bal += day_pnl
            peak = max(peak, bal)
            if peak - bal > cfg["trailing_dd"]:
                broke = True
                break
        if broke:
            blowups += 1
        end_balances[r] = bal

    blowup_prob = blowups / n_runs
    return PropResult(
        account=account_name,
        n_runs=n_runs,
        blowup_probability=round(blowup_prob, 4),
        prop_survival_score=round(1.0 - blowup_prob, 4),
        end_balance_p10=round(float(np.percentile(end_balances, 10)), 2),
        end_balance_p50=round(float(np.percentile(end_balances, 50)), 2),
        end_balance_p90=round(float(np.percentile(end_balances, 90)), 2),
        passes=blowup_prob <= 0.20 and float(np.percentile(end_balances, 50)) > cfg["balance"],
    )


def simulate_all(trades: list[Trade], **kwargs) -> dict:
    out = {}
    for name in PROP_ACCOUNTS:
        r = simulate_account(trades, name, **kwargs)
        out[name] = {
            "blowup_probability": r.blowup_probability,
            "prop_survival_score": r.prop_survival_score,
            "end_balance_p10": r.end_balance_p10,
            "end_balance_p50": r.end_balance_p50,
            "end_balance_p90": r.end_balance_p90,
            f"passes_{name}": r.passes,
        }
    return out
