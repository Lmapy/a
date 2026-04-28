"""Funded-account payout simulation (Phase 7 hardening).

Same machinery as challenge.py but the goal is to reach `payout_target`
over `payout_min_days`. Used to estimate first-payout probability for
strategies that pass the challenge phase.

Day-block bootstrap; no per-trade leakage; Wilson CI on the
first-payout probability. Risk sizing goes through the new RiskModel
that does NOT consume `trade_pnl_price`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prop_challenge.accounts import AccountSpec, dollar_per_price_unit
from prop_challenge.challenge import (
    _normalise_trades,
    _sample_day_blocks,
    _stop_distance,
    _atr_pre_entry,
    N_RUNS_EXPLORATION,
)
from prop_challenge.drawdown import DrawdownState
from prop_challenge.lockout import DailyRules, DayState, admit_trade, update_day
from prop_challenge.risk import RiskModel
from prop_challenge.stats import wilson_ci, bootstrap_median_ci


@dataclass
class PayoutResult:
    n_runs: int
    first_payout_probability: float
    first_payout_probability_ci: tuple[float, float]
    blowup_before_payout_probability: float
    blowup_before_payout_probability_ci: tuple[float, float]
    consistency_breach_rate: float
    median_days_to_payout: float | None
    median_days_to_payout_ci: tuple[float, float] | None


def _replay_for_payout(days: list[tuple], account: AccountSpec,
                        risk: RiskModel, rules: DailyRules,
                        instrument: str, max_days: int) -> dict:
    state = DrawdownState.fresh(account)
    last_loss = False
    max_balance = account.starting_balance
    dpp = dollar_per_price_unit(instrument)

    for day_idx, (_, day_df) in enumerate(days):
        if day_idx >= max_days:
            break
        day = DayState()
        for _, row in day_df.iterrows():
            ts = pd.Timestamp(row["entry_time"])
            if not admit_trade(rules, day, ts):
                continue
            equity_high = state.balance >= max_balance
            sd = _stop_distance(row)
            atr = _atr_pre_entry(row)
            contracts = risk.size(
                balance=state.balance,
                starting_balance=state.starting_balance,
                max_loss=account.max_loss,
                last_trade_loss=last_loss,
                equity_high=equity_high,
                stop_distance_price=sd,
                atr_pre_entry=atr,
            )
            dollar_pnl = float(row["pnl"]) * dpp * contracts - 0.5 * contracts
            state.apply_trade(dollar_pnl, day_idx)
            update_day(rules, day, dollar_pnl)
            last_loss = dollar_pnl < 0
            max_balance = max(max_balance, state.balance)
            if not state.alive():
                return {"outcome": "blowup", "days": day_idx + 1,
                        "end_balance": state.balance,
                        "consistency_breach": False}
        state.end_of_day(day_idx)
        if not state.alive():
            return {"outcome": "blowup", "days": day_idx + 1,
                    "end_balance": state.balance,
                    "consistency_breach": False}
        if (state.balance - state.starting_balance) >= account.payout_target \
                and state.days_traded >= account.payout_min_days:
            return {"outcome": "payout", "days": day_idx + 1,
                    "end_balance": state.balance,
                    "consistency_breach": state.consistency_breached()}

    return {"outcome": "timeout",
            "days": min(len(days), max_days),
            "end_balance": state.balance,
            "consistency_breach": False}


def simulate_payout(trades_df: pd.DataFrame,
                    account: AccountSpec,
                    risk: RiskModel,
                    rules: DailyRules,
                    *,
                    n_runs: int = N_RUNS_EXPLORATION,
                    instrument: str = "MGC",
                    block_days: int = 5,
                    max_days: int | None = 60,
                    seed: int = 33) -> PayoutResult:
    # untimed evaluations: cap at 60 days for MC tractability.
    if max_days is None:
        max_days = 60
    if trades_df.empty:
        return PayoutResult(
            n_runs=0,
            first_payout_probability=0.0,
            first_payout_probability_ci=(0.0, 1.0),
            blowup_before_payout_probability=1.0,
            blowup_before_payout_probability_ci=(0.0, 1.0),
            consistency_breach_rate=0.0,
            median_days_to_payout=None,
            median_days_to_payout_ci=None,
        )

    df = _normalise_trades(trades_df)
    all_days = list(df.groupby("date"))
    rng = np.random.default_rng(seed)

    n_pay = 0
    n_blow = 0
    n_cons = 0
    days_to_payout: list[int] = []
    for _ in range(n_runs):
        sampled = _sample_day_blocks(all_days, block_days, max_days, rng)
        out = _replay_for_payout(sampled, account, risk, rules,
                                  instrument, max_days)
        if out["outcome"] == "payout":
            n_pay += 1
            days_to_payout.append(out["days"])
            if out["consistency_breach"]:
                n_cons += 1
        elif out["outcome"] == "blowup":
            n_blow += 1

    pp, pp_lo, pp_hi = wilson_ci(n_pay, n_runs)
    bp, bp_lo, bp_hi = wilson_ci(n_blow, n_runs)

    if days_to_payout:
        arr = np.array(days_to_payout, dtype=float)
        med, lo, hi = bootstrap_median_ci(arr, n_runs=min(2000, max(200, n_runs)))
        med_ci = (round(lo, 1), round(hi, 1))
    else:
        med = None
        med_ci = None

    return PayoutResult(
        n_runs=n_runs,
        first_payout_probability=round(pp, 4),
        first_payout_probability_ci=(round(pp_lo, 4), round(pp_hi, 4)),
        blowup_before_payout_probability=round(bp, 4),
        blowup_before_payout_probability_ci=(round(bp_lo, 4), round(bp_hi, 4)),
        consistency_breach_rate=round((n_cons / n_pay) if n_pay else 0.0, 4),
        median_days_to_payout=(round(float(med), 1) if med is not None else None),
        median_days_to_payout_ci=med_ci,
    )
