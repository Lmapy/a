"""Load prop-firm account definitions from config/prop_accounts.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "prop_accounts.json"


@dataclass
class AccountSpec:
    name: str
    firm: str
    starting_balance: float
    profit_target: float
    daily_loss_limit: float
    max_loss: float
    trailing_drawdown: float
    drawdown_type: str           # static | eod_trailing | intraday_trailing
    max_contracts: int
    minimum_trading_days: int
    consistency_rule_percent: float   # max % of profit allowed from one day
    payout_target: float
    payout_min_days: int
    max_challenge_days: int


def load_all() -> dict[str, AccountSpec]:
    raw = json.loads(CONFIG.read_text())
    out: dict[str, AccountSpec] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        out[k] = AccountSpec(
            name=k,
            firm=v["firm"],
            starting_balance=float(v["starting_balance"]),
            profit_target=float(v["profit_target"]),
            daily_loss_limit=float(v["daily_loss_limit"]),
            max_loss=float(v["max_loss"]),
            trailing_drawdown=float(v["trailing_drawdown"]),
            drawdown_type=v["drawdown_type"],
            max_contracts=int(v["max_contracts"]),
            minimum_trading_days=int(v["minimum_trading_days"]),
            consistency_rule_percent=float(v["consistency_rule_percent"]),
            payout_target=float(v["payout_target"]),
            payout_min_days=int(v["payout_min_days"]),
            max_challenge_days=int(v["max_challenge_days"]),
        )
    return out


def dollar_per_price_unit(instrument: str = "MGC") -> float:
    raw = json.loads(CONFIG.read_text())
    return float(raw["_meta"]["instrument_dollar_per_price_unit"][instrument])
