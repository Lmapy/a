"""Challenge simulator (Phase 7 hardening).

Two execution modes:

  1. run_chronological_replay
     Replay the actual trade ledger in timestamp order. Trades are
     grouped by their REAL calendar day (UTC). Daily lockouts,
     daily-loss limits, trailing drawdown and consistency rule fire
     in chronological order on actual days. This is deterministic
     for a given (trades, account, risk, rules) tuple.

  2. run_challenge
     Monte Carlo. Resamples DAYS (not individual trades) to preserve
     intra-day clustering of wins/losses. Each MC draw selects a
     contiguous block of historical trading days and replays them.
     Reports point estimates plus Wilson 95% CIs for pass / blowup /
     timeout / consistency_breach probabilities.

The position-sizing path here goes through the new RiskModel signature
that does NOT consume `trade_pnl_price`. We pass `stop_distance_price`
(known at fill time) so dollar-risk and pct-DD-buffer sizing models
can operate without future leakage. If the trade row has no
`stop_distance_price` column we leave it None and the risk model
falls back to base contracts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from prop_challenge.accounts import AccountSpec, dollar_per_price_unit
from prop_challenge.drawdown import DrawdownState
from prop_challenge.lockout import DailyRules, DayState, admit_trade, update_day
from prop_challenge.risk import RiskModel
from prop_challenge.stats import wilson_ci, bootstrap_median_ci


# Default Monte Carlo counts. Final certification should override.
N_RUNS_EXPLORATION = 500
N_RUNS_FINAL = 5000


@dataclass
class ChallengeResult:
    n_runs: int
    pass_probability: float
    pass_probability_ci: tuple[float, float]      # (lower, upper) Wilson 95%
    blowup_probability: float
    blowup_probability_ci: tuple[float, float]
    timeout_probability: float
    timeout_probability_ci: tuple[float, float]
    consistency_breach_rate: float
    median_days_to_pass: float | None
    median_days_to_pass_ci: tuple[float, float] | None
    median_end_balance: float
    median_breach_day: float | None
    breach_reason_histogram: dict[str, int]
    mode: str = "monte_carlo"           # monte_carlo | chronological
    extras: dict = field(default_factory=dict)


# ---------------- chronological replay ----------------

def _normalise_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Add `date` column (UTC normalised) and sort by entry_time. Idempotent."""
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["date"] = df["entry_time"].dt.normalize()
    return df


def _stop_distance(row: pd.Series) -> float | None:
    """Best-effort stop distance from a trade row. Looks at, in order:
      stop_distance_price (preferred, in price units)
      stop_price + entry        (compute from columns)
      None                       (sizing falls back to base contracts)
    """
    if "stop_distance_price" in row and pd.notna(row["stop_distance_price"]) and float(row["stop_distance_price"]) > 0:
        return float(row["stop_distance_price"])
    if "stop_price" in row and pd.notna(row["stop_price"]) and "entry" in row:
        try:
            sd = abs(float(row["entry"]) - float(row["stop_price"]))
            return sd if sd > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def _atr_pre_entry(row: pd.Series) -> float | None:
    """ATR computed on completed bars before this entry, if the runner
    populated the column. Otherwise None."""
    if "atr_pre_entry" in row and pd.notna(row["atr_pre_entry"]) and float(row["atr_pre_entry"]) > 0:
        return float(row["atr_pre_entry"])
    return None


def _replay_days(days: list[tuple], account: AccountSpec, risk: RiskModel,
                 rules: DailyRules, instrument: str,
                 max_days: int) -> dict:
    """Walk `days` (list of (date, day_dataframe)) chronologically and
    apply rules. Returns the same per-path summary shape `_simulate_one`
    used to return."""
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
            update_day(rules, day, dollar_pnl, ts=ts)
            last_loss = dollar_pnl < 0
            max_balance = max(max_balance, state.balance)
            if not state.alive():
                return {"outcome": "blowup", "breach": state.last_breach,
                        "days": day_idx + 1, "end_balance": state.balance,
                        "consistency_breach": False}

        state.end_of_day(day_idx)
        if not state.alive():
            return {"outcome": "blowup", "breach": state.last_breach,
                    "days": day_idx + 1, "end_balance": state.balance,
                    "consistency_breach": False}
        if state.reached_target() and state.days_traded >= account.minimum_trading_days:
            return {"outcome": "pass", "breach": None,
                    "days": day_idx + 1, "end_balance": state.balance,
                    "consistency_breach": state.consistency_breached()}
    return {"outcome": "timeout", "breach": None,
            "days": min(len(days), max_days), "end_balance": state.balance,
            "consistency_breach": False}


def run_chronological_replay(trades_df: pd.DataFrame,
                              account: AccountSpec,
                              risk: RiskModel,
                              rules: DailyRules,
                              *,
                              instrument: str = "MGC") -> dict:
    """Deterministic single-path replay of the actual ledger."""
    if trades_df.empty:
        return {"outcome": "no_trades", "breach": "no_trades", "days": 0,
                "end_balance": account.starting_balance, "consistency_breach": False}
    df = _normalise_trades(trades_df)
    days = list(df.groupby("date"))
    # `max_challenge_days` may be None for untimed evaluations; cap at
    # the actual day count plus a generous margin so the replay always
    # finishes within the historical ledger.
    cap = account.max_challenge_days if account.max_challenge_days is not None \
        else max(len(days), 9999)
    return _replay_days(days, account, risk, rules, instrument, cap)


# ---------------- day-block Monte Carlo ----------------

def _sample_day_blocks(all_days: list[tuple], block_days: int,
                       n_days: int, rng: np.random.Generator) -> list[tuple]:
    """Sample `n_days` of trading days as contiguous blocks of length
    `block_days`. Wraps around at the end (circular block bootstrap)."""
    if not all_days:
        return []
    out: list[tuple] = []
    while len(out) < n_days:
        start = int(rng.integers(0, len(all_days)))
        take = min(block_days, n_days - len(out))
        for k in range(take):
            out.append(all_days[(start + k) % len(all_days)])
    return out


def run_challenge(trades_df: pd.DataFrame,
                  account: AccountSpec,
                  risk: RiskModel,
                  rules: DailyRules,
                  *,
                  n_runs: int = N_RUNS_EXPLORATION,
                  instrument: str = "MGC",
                  block_days: int = 5,
                  seed: int = 11) -> ChallengeResult:
    """Day-level block-bootstrap Monte Carlo.

    Sampling DAYS (not individual trades) preserves intra-day
    clustering — winning days are still more likely to follow winning
    days, and the daily loss limit / consistency rule see realistic
    aggregates instead of synthetic 4-12-trade randomly-grouped chunks.
    """
    if trades_df.empty:
        empty = (0.0, 0.0, 1.0)
        return ChallengeResult(
            n_runs=0,
            pass_probability=0.0, pass_probability_ci=(0.0, 1.0),
            blowup_probability=0.0, blowup_probability_ci=(0.0, 1.0),
            timeout_probability=0.0, timeout_probability_ci=(0.0, 1.0),
            consistency_breach_rate=0.0,
            median_days_to_pass=None, median_days_to_pass_ci=None,
            median_end_balance=account.starting_balance,
            median_breach_day=None,
            breach_reason_histogram={}, mode="monte_carlo")

    df = _normalise_trades(trades_df)
    all_days = list(df.groupby("date"))
    # `max_challenge_days` may be None for untimed evaluations. Use a
    # default of 60 days for MC sampling so we don't accidentally allow
    # an unbounded path; the chronological replay above keeps the full
    # ledger length when the firm imposes no cap.
    target_days = account.max_challenge_days if account.max_challenge_days is not None else 60

    rng = np.random.default_rng(seed)
    outcomes: list[dict] = []
    for _ in range(n_runs):
        sampled = _sample_day_blocks(all_days, block_days, target_days, rng)
        outcomes.append(_replay_days(sampled, account, risk, rules,
                                      instrument, target_days))

    n_pass = sum(1 for o in outcomes if o["outcome"] == "pass")
    n_blow = sum(1 for o in outcomes if o["outcome"] == "blowup")
    n_time = sum(1 for o in outcomes if o["outcome"] == "timeout")
    cons_breach = sum(1 for o in outcomes if o["outcome"] == "pass" and o["consistency_breach"])
    pass_days = np.array([o["days"] for o in outcomes if o["outcome"] == "pass"], dtype=float)
    blow_days = [o["days"] for o in outcomes if o["outcome"] == "blowup"]
    end_balances = [o["end_balance"] for o in outcomes]

    pp, pp_lo, pp_hi = wilson_ci(n_pass, n_runs)
    bp, bp_lo, bp_hi = wilson_ci(n_blow, n_runs)
    tp, tp_lo, tp_hi = wilson_ci(n_time, n_runs)

    if len(pass_days):
        med, lo, hi = bootstrap_median_ci(pass_days, n_runs=min(2000, max(200, n_runs)))
        med_ci = (round(lo, 1), round(hi, 1))
    else:
        med = None
        med_ci = None

    breach_hist: dict[str, int] = {}
    for o in outcomes:
        if o["breach"]:
            breach_hist[o["breach"]] = breach_hist.get(o["breach"], 0) + 1

    return ChallengeResult(
        n_runs=n_runs,
        pass_probability=round(pp, 4),
        pass_probability_ci=(round(pp_lo, 4), round(pp_hi, 4)),
        blowup_probability=round(bp, 4),
        blowup_probability_ci=(round(bp_lo, 4), round(bp_hi, 4)),
        timeout_probability=round(tp, 4),
        timeout_probability_ci=(round(tp_lo, 4), round(tp_hi, 4)),
        consistency_breach_rate=round((cons_breach / n_pass) if n_pass else 0.0, 4),
        median_days_to_pass=(round(float(med), 1) if med is not None else None),
        median_days_to_pass_ci=med_ci,
        median_end_balance=round(float(np.median(end_balances)), 2),
        median_breach_day=(round(float(np.median(blow_days)), 1) if blow_days else None),
        breach_reason_histogram=breach_hist,
        mode="monte_carlo",
        extras={"block_days": block_days, "n_historical_days": len(all_days)},
    )
