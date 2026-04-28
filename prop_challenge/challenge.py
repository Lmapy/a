"""Challenge Monte-Carlo simulator.

Given a list of historical trades for a strategy, an account spec, a
risk model, and a daily-rule set, run N MC paths through the prop
challenge phase and return aggregate statistics:

  pass_probability         share of paths that hit profit_target
                            without breaching any rule and met the
                            minimum_trading_days
  blowup_probability       share of paths that hit any breach
  timeout_probability      share of paths that ran out of days
  consistency_breach_rate  share of passing-by-balance paths that
                            also breached the consistency rule
  median_days_to_pass      among passing paths
  median_end_balance       across all paths

Trades are sampled with stationary block-bootstrap from the provided
trade ledger, then grouped into trading days using a configurable
"trades per day" bootstrap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prop_challenge.accounts import AccountSpec, dollar_per_price_unit
from prop_challenge.drawdown import DrawdownState
from prop_challenge.lockout import DailyRules, DayState, admit_trade, update_day
from prop_challenge.risk import RiskModel


@dataclass
class ChallengeResult:
    n_runs: int
    pass_probability: float
    blowup_probability: float
    timeout_probability: float
    consistency_breach_rate: float
    median_days_to_pass: float | None
    median_end_balance: float
    median_breach_day: float | None
    breach_reason_histogram: dict[str, int]


# ---------- block-bootstrap ----------

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


# ---------- per-path sim ----------

def _simulate_one(
    trades_df: pd.DataFrame,
    account: AccountSpec,
    risk: RiskModel,
    rules: DailyRules,
    instrument: str,
    rng: np.random.Generator,
    block: int,
    trades_per_day_lo: int,
    trades_per_day_hi: int,
) -> dict:
    """Run one MC path; return per-path summary."""
    if trades_df.empty:
        return {"outcome": "blowup", "breach": "no_trades", "days": 0,
                "end_balance": account.starting_balance, "consistency_breach": False}

    pnl_price = trades_df["pnl"].values
    times = pd.to_datetime(trades_df["entry_time"], utc=True).values
    dpp = dollar_per_price_unit(instrument)

    state = DrawdownState.fresh(account)
    last_loss = False
    max_balance = account.starting_balance
    day_idx = 0

    # Loop day-by-day until pass / blowup / timeout
    while day_idx < account.max_challenge_days:
        # how many trades today?
        n_trades_today = int(rng.integers(trades_per_day_lo, trades_per_day_hi + 1))
        if n_trades_today == 0:
            day_idx += 1
            continue
        # bootstrap that many trade returns
        seq = _block_bootstrap(pnl_price, block=block, n=n_trades_today, rng=rng)
        # synthesise per-trade timestamps spread across the day (for session filter)
        # For simplicity we sample trade times from the historical ledger uniformly.
        ts_choice = rng.choice(len(times), size=n_trades_today, replace=True)
        sampled_ts = pd.to_datetime(times[ts_choice], utc=True)

        day = DayState()
        for j in range(n_trades_today):
            ts = sampled_ts[j]
            if not admit_trade(rules, day, ts):
                continue
            # size the trade
            equity_high = state.balance >= max_balance
            contracts = risk.size(
                balance=state.balance,
                starting_balance=state.starting_balance,
                max_loss=account.max_loss,
                last_trade_loss=last_loss,
                equity_high=equity_high,
                trade_pnl_price=float(seq[j]),
            )
            dollar_pnl = float(seq[j]) * dpp * contracts
            # Apply slippage proxy: a tiny adverse cost per contract per trade.
            dollar_pnl -= 0.5 * contracts
            state.apply_trade(dollar_pnl, day_idx)
            update_day(rules, day, dollar_pnl)
            last_loss = dollar_pnl < 0
            max_balance = max(max_balance, state.balance)
            if not state.alive():
                break

        state.end_of_day(day_idx)

        if not state.alive():
            return {
                "outcome": "blowup",
                "breach": state.last_breach,
                "days": day_idx + 1,
                "end_balance": state.balance,
                "consistency_breach": False,
            }

        if state.reached_target() and state.days_traded >= account.minimum_trading_days:
            return {
                "outcome": "pass",
                "breach": None,
                "days": day_idx + 1,
                "end_balance": state.balance,
                "consistency_breach": state.consistency_breached(),
            }
        day_idx += 1

    return {
        "outcome": "timeout",
        "breach": None,
        "days": account.max_challenge_days,
        "end_balance": state.balance,
        "consistency_breach": False,
    }


# ---------- top-level ----------

def run_challenge(
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
    seed: int = 11,
) -> ChallengeResult:
    rng = np.random.default_rng(seed)
    outcomes: list[dict] = []
    for _ in range(n_runs):
        outcomes.append(_simulate_one(
            trades_df, account, risk, rules,
            instrument=instrument, rng=rng, block=block,
            trades_per_day_lo=trades_per_day_lo, trades_per_day_hi=trades_per_day_hi,
        ))
    n_pass = sum(1 for o in outcomes if o["outcome"] == "pass")
    n_blow = sum(1 for o in outcomes if o["outcome"] == "blowup")
    n_time = sum(1 for o in outcomes if o["outcome"] == "timeout")
    cons_breach = sum(1 for o in outcomes if o["outcome"] == "pass" and o["consistency_breach"])
    pass_days = [o["days"] for o in outcomes if o["outcome"] == "pass"]
    blow_days = [o["days"] for o in outcomes if o["outcome"] == "blowup"]
    end_balances = [o["end_balance"] for o in outcomes]
    breach_hist: dict[str, int] = {}
    for o in outcomes:
        if o["breach"]:
            breach_hist[o["breach"]] = breach_hist.get(o["breach"], 0) + 1
    return ChallengeResult(
        n_runs=n_runs,
        pass_probability=n_pass / n_runs,
        blowup_probability=n_blow / n_runs,
        timeout_probability=n_time / n_runs,
        consistency_breach_rate=(cons_breach / n_pass) if n_pass else 0.0,
        median_days_to_pass=(float(np.median(pass_days)) if pass_days else None),
        median_end_balance=float(np.median(end_balances)),
        median_breach_day=(float(np.median(blow_days)) if blow_days else None),
        breach_reason_histogram=breach_hist,
    )
