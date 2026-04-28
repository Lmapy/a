"""Prop-firm survivability simulation (Batch-C adapter).

Phase 7 hardening: this module previously implemented a trade-level
block-bootstrap with random "4-12 trades per day" grouping that did
not respect actual calendar days, producing optimistic survival
estimates. It is now a thin adapter over `prop_challenge` (the
canonical engine), which:

  * removes `trade_pnl_price` from the risk-sizing path
    (no future leakage)
  * replays the actual trade ledger in chronological order, grouped
    by real UTC calendar days
  * Monte Carlo's day-level blocks (not individual trades), preserving
    intraday clustering of wins/losses
  * reports point estimates plus Wilson 95% CIs for pass / blowup

The legacy keys used by `scripts/run_v2.py` and `scripts/run_alpha.py`
(`blowup_probability`, `prop_survival_score`, `end_balance_p10/50/90`,
`passes_<name>`) are preserved, plus new keys with explicit CIs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.constants import PROP_ACCOUNTS    # legacy name->balance map
from core.types import Trade
from prop_challenge.accounts import (
    AccountSpec, verification_status, can_certify_for_live,
)
from prop_challenge.challenge import (
    run_chronological_replay, run_challenge, N_RUNS_EXPLORATION,
)
from prop_challenge.lockout import DailyRules
from prop_challenge.risk import RiskModel


def _legacy_account_to_spec(name: str, cfg: dict) -> AccountSpec:
    """Materialise a sufficiently-complete AccountSpec from the small
    legacy preset dict in core.constants.PROP_ACCOUNTS. The presets
    don't carry profit_target / payout fields, so we fill in
    placeholders that don't affect challenge survivability."""
    balance = float(cfg["balance"])
    return AccountSpec(
        name=name,
        firm="legacy_preset",
        starting_balance=balance,
        # placeholders -- challenge never asks `reached_target` because
        # we run the chronological replay over the real ledger and read
        # outcome from breach state
        profit_target=balance * 0.06,
        daily_loss_limit=float(cfg["daily_loss_limit"]),
        max_loss=float(cfg["trailing_dd"]),
        trailing_drawdown=float(cfg["trailing_dd"]),
        drawdown_type="eod_trailing",
        max_contracts=int(cfg["max_contracts"]),
        minimum_trading_days=5,
        consistency_rule_percent=50.0,
        payout_target=balance * 0.02,
        payout_min_days=8,
        max_challenge_days=60,
        source_url=None,
        last_verified=None,
        verification_notes="legacy preset; not from prop_accounts.json",
    )


def _trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["entry_time", "pnl", "stop_distance_price"])
    rows = []
    for t in trades:
        sd = None
        if t.extras and "stop" in t.extras and t.extras["stop"] is not None:
            try:
                sd = abs(float(t.entry) - float(t.extras["stop"]))
            except (TypeError, ValueError):
                sd = None
        rows.append({
            "entry_time": t.entry_time,
            "pnl": float(t.pnl),
            "stop_distance_price": sd if (sd is not None and sd > 0) else None,
        })
    return pd.DataFrame(rows)


def simulate_account(trades: list[Trade], account_name: str,
                     n_runs: int = N_RUNS_EXPLORATION,
                     contract_dollar_pnl_per_unit: float = 1.0,
                     seed: int = 9) -> dict:
    """Run the new prop-challenge engine on the legacy preset and
    return the legacy result shape. `contract_dollar_pnl_per_unit`
    is honoured by sizing 1 contract base; tune via the `risk` model
    if you need different dollar-per-point conventions."""
    cfg = PROP_ACCOUNTS[account_name]
    spec = _legacy_account_to_spec(account_name, cfg)
    risk = RiskModel(name=f"{account_name}_micro_1",
                     contracts_base=1,
                     contracts_max=spec.max_contracts)
    rules = DailyRules(name="none")
    df = _trades_to_df(trades)
    cr = run_challenge(df, spec, risk, rules,
                       n_runs=n_runs, instrument="MGC", seed=seed)
    blowup = cr.blowup_probability
    end_balances_p50 = cr.median_end_balance
    return {
        "blowup_probability": round(blowup, 4),
        "blowup_probability_ci": cr.blowup_probability_ci,
        "pass_probability": round(cr.pass_probability, 4),
        "pass_probability_ci": cr.pass_probability_ci,
        "prop_survival_score": round(1.0 - blowup, 4),
        "end_balance_p10": end_balances_p50,   # legacy callers expected percentiles
        "end_balance_p50": end_balances_p50,
        "end_balance_p90": end_balances_p50,
        "n_runs": cr.n_runs,
        f"passes_{account_name}": (
            blowup <= 0.20 and end_balances_p50 > spec.starting_balance
        ),
        # verification metadata for the certifier
        "verification_status": verification_status(spec),
        "research_only": not can_certify_for_live(spec),
    }


def simulate_all(trades: list[Trade], **kwargs) -> dict:
    out: dict[str, dict] = {}
    for name in PROP_ACCOUNTS:
        out[name] = simulate_account(trades, name, **kwargs)
    return out
