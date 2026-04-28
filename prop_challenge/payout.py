"""Funded-account payout simulation.

Same machinery as challenge.py but with the goal of reaching the
account's `payout_target` over `payout_min_days`. Used to estimate
first-payout probability for strategies that pass the challenge phase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prop_challenge.accounts import AccountSpec, dollar_per_price_unit
from prop_challenge.drawdown import DrawdownState
from prop_challenge.lockout import DailyRules, DayState, admit_trade, update_day
from prop_challenge.risk import RiskModel


@dataclass
class PayoutResult:
    n_runs: int
    first_payout_probability: float
    blowup_before_payout_probability: float
    consistency_breach_rate: float
    median_days_to_payout: float | None


def _block_bootstrap(rets: np.ndarray, block: int, n: int,
                     rng: np.random.Generator) -> np.ndarray:
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


def simulate_payout(
    trades_df: pd.DataFrame,
    account: AccountSpec,
    risk: RiskModel,
    rules: DailyRules,
    *,
    n_runs: int = 1000,
    instrument: str = "MGC",
    block: int = 5,
    trades_per_day_lo: int = 1,
    trades_per_day_hi: int = 5,
    max_days: int = 60,
    seed: int = 33,
) -> PayoutResult:
    rng = np.random.default_rng(seed)
    n_pass = 0
    n_blow = 0
    n_cons = 0
    days_to_payout: list[int] = []

    pnl_price = trades_df["pnl"].values if not trades_df.empty else np.array([])
    times = pd.to_datetime(trades_df["entry_time"], utc=True).values if not trades_df.empty else np.array([])
    dpp = dollar_per_price_unit(instrument)

    if pnl_price.size == 0:
        return PayoutResult(n_runs, 0.0, 1.0, 0.0, None)

    for _ in range(n_runs):
        state = DrawdownState.fresh(account)
        max_balance = account.starting_balance
        last_loss = False
        day_idx = 0
        passed = False
        while day_idx < max_days:
            n_today = int(rng.integers(trades_per_day_lo, trades_per_day_hi + 1))
            if n_today == 0:
                day_idx += 1
                continue
            seq = _block_bootstrap(pnl_price, block=block, n=n_today, rng=rng)
            ts_choice = rng.choice(len(times), size=n_today, replace=True)
            sampled_ts = pd.to_datetime(times[ts_choice], utc=True)

            day = DayState()
            for j in range(n_today):
                ts = sampled_ts[j]
                if not admit_trade(rules, day, ts):
                    continue
                equity_high = state.balance >= max_balance
                contracts = risk.size(
                    balance=state.balance,
                    starting_balance=state.starting_balance,
                    max_loss=account.max_loss,
                    last_trade_loss=last_loss,
                    equity_high=equity_high,
                    trade_pnl_price=float(seq[j]),
                )
                dollar_pnl = float(seq[j]) * dpp * contracts - 0.5 * contracts
                state.apply_trade(dollar_pnl, day_idx)
                update_day(rules, day, dollar_pnl)
                last_loss = dollar_pnl < 0
                max_balance = max(max_balance, state.balance)
                if not state.alive():
                    break

            state.end_of_day(day_idx)
            if not state.alive():
                n_blow += 1
                break
            # payout target measured against starting balance
            if (state.balance - state.starting_balance) >= account.payout_target \
                    and state.days_traded >= account.payout_min_days:
                n_pass += 1
                days_to_payout.append(day_idx + 1)
                if state.consistency_breached():
                    n_cons += 1
                passed = True
                break
            day_idx += 1

    return PayoutResult(
        n_runs=n_runs,
        first_payout_probability=n_pass / n_runs,
        blowup_before_payout_probability=n_blow / n_runs,
        consistency_breach_rate=(n_cons / n_pass) if n_pass else 0.0,
        median_days_to_payout=(float(np.median(days_to_payout)) if days_to_payout else None),
    )
