"""Bridge CBR scalp trade ledgers into the prop-firm passing engine.

The Batch C `prop_challenge.challenge.run_chronological_replay`
(and `run_challenge` MC) consume a DataFrame with these columns:

    entry_time             tz-aware UTC timestamp
    pnl                    realised PnL in PRICE units (not dollars)
    stop_distance_price    optional, |entry - stop| in price units

Our CBR engine writes `trades.csv` with `entry_time`, `entry_price`,
`stop_price`, `pnl` (in dollars, computed via tick_value), `r_result`,
direction, etc. We need to rewrite into the prop-challenge shape:

    entry_time         <- trades.entry_time
    pnl (price units)  <- (exit_price - entry_price) * direction
    stop_distance_price <- |entry_price - stop_price|

Once the DF is in that shape, the prop-firm Wilson-CI machinery from
Batch C applies unchanged. This module returns the DataFrame and
optionally the prop-challenge result for a default account / risk /
rules combination.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from prop_challenge.accounts import AccountSpec, load_all as load_accounts
from prop_challenge.challenge import (
    N_RUNS_EXPLORATION, run_challenge, run_chronological_replay,
)
from prop_challenge.lockout import DailyRules
from prop_challenge.risk import RiskModel
from prop_challenge.score import passes_cert
from prop_challenge.payout import simulate_payout


def load_cbr_trades(trades_csv: Path | str) -> pd.DataFrame:
    """Load `trades.csv` produced by the CBR engine and reshape into
    the prop-challenge format."""
    df = pd.read_csv(trades_csv)
    if df.empty:
        return df.assign(entry_time=pd.Series(dtype="datetime64[ns, UTC]"),
                          pnl=pd.Series(dtype=float),
                          stop_distance_price=pd.Series(dtype=float))
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["stop_distance_price"] = (df["entry_price"] - df["stop_price"]).abs()
    # the CBR `pnl` is dollar-PnL; the prop sim expects PRICE-unit PnL
    # so a per-trade R-multiple sized by the prop sim's RiskModel
    # turns into a dollar number consistent with the account spec.
    # We rebuild price-unit pnl as direction * (exit - entry).
    df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["direction"]
    return df[["entry_time", "pnl", "stop_distance_price"]].sort_values(
        "entry_time").reset_index(drop=True)


def run_prop_replay(trades_csv: Path | str, *,
                     account_name: str = "topstep_50k",
                     n_runs: int = N_RUNS_EXPLORATION,
                     instrument: str = "MGC",
                     output_dir: Path | None = None) -> dict:
    """Run BOTH chronological replay AND day-block MC on the CBR
    ledger; return the combined report."""
    trades_df = load_cbr_trades(trades_csv)
    accounts = load_accounts()
    if account_name not in accounts:
        raise KeyError(
            f"account {account_name!r} not in prop_accounts.json; "
            f"choose from {sorted(accounts.keys())}")
    account = accounts[account_name]
    risk = RiskModel(name="micro_1", contracts_base=1,
                     contracts_max=account.max_contracts)
    rules = DailyRules(name="none")

    chrono = run_chronological_replay(trades_df, account, risk, rules,
                                        instrument=instrument)
    cr = run_challenge(trades_df, account, risk, rules,
                        n_runs=n_runs, instrument=instrument)
    payout = simulate_payout(trades_df, account, risk, rules,
                              n_runs=n_runs, instrument=instrument)
    cert_ok, cert_fails = passes_cert(cr, payout)

    payload = {
        "n_trades": int(len(trades_df)),
        "account": account_name,
        "instrument": instrument,
        "chronological_replay": chrono,
        "monte_carlo": {
            "n_runs": cr.n_runs,
            "pass_probability": cr.pass_probability,
            "pass_probability_ci": list(cr.pass_probability_ci),
            "blowup_probability": cr.blowup_probability,
            "blowup_probability_ci": list(cr.blowup_probability_ci),
            "median_days_to_pass": cr.median_days_to_pass,
            "median_end_balance": cr.median_end_balance,
            "breach_reasons": cr.breach_reason_histogram,
        },
        "payout": {
            "n_runs": payout.n_runs,
            "first_payout_probability": payout.first_payout_probability,
            "first_payout_probability_ci": list(payout.first_payout_probability_ci),
            "blowup_before_payout_probability": payout.blowup_before_payout_probability,
            "median_days_to_payout": payout.median_days_to_payout,
        },
        "cert": {
            "passes": cert_ok,
            "failures": cert_fails,
        },
    }
    if output_dir is not None:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prop_replay.json").write_text(
            json.dumps(payload, indent=2, default=str))
    return payload
