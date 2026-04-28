"""Load + validate prop-firm account definitions from config/prop_accounts.json.

Phase 8 hardening:
  * `AccountSpec` carries verification metadata (`source_url`,
    `last_verified`, `notes`).
  * `verification_status(spec, today)` returns one of:
      "verified"       -- last_verified within 90 days of `today`
      "stale"          -- last_verified older than 90 days
      "unverified"     -- last_verified is None / missing
      "synthetic"      -- source_url == "synthetic" (test fixtures)
  * `validate_schema(raw)` rejects accounts missing required keys; this
    runs at load time so a bad config fails loudly before any sim.
  * `load_all` returns the parsed + validated spec dict.
  * The certifier / runners can read `verification_status()` and mark
    results research_only when not verified.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "prop_accounts.json"

REQUIRED_FIELDS = (
    "firm", "starting_balance", "profit_target", "daily_loss_limit",
    "max_loss", "trailing_drawdown", "drawdown_type",
    "max_contracts", "minimum_trading_days", "consistency_rule_percent",
    "payout_target", "payout_min_days", "max_challenge_days",
)
DRAWDOWN_TYPES = {"static", "eod_trailing", "intraday_trailing"}
VERIFICATION_STALE_DAYS = 90


@dataclass
class AccountSpec:
    name: str
    firm: str
    starting_balance: float
    profit_target: float
    daily_loss_limit: float | None       # None = no DLL (e.g. MFFU 2026)
    max_loss: float
    trailing_drawdown: float
    drawdown_type: str
    max_contracts: int
    minimum_trading_days: int
    consistency_rule_percent: float
    payout_target: float
    payout_min_days: int
    max_challenge_days: int | None       # None = untimed evaluation
    source_url: str | None = None
    last_verified: str | None = None
    verification_notes: str | None = None
    extras: dict = field(default_factory=dict)


def validate_schema(raw: dict) -> None:
    """Fail loudly on a malformed config.

    Raises ValueError with a clear message on the first problem found.
    """
    if not isinstance(raw, dict):
        raise ValueError("prop_accounts.json must be a JSON object at the top level")
    if "_meta" not in raw:
        raise ValueError("prop_accounts.json missing `_meta` block")
    meta = raw["_meta"]
    if "instrument_dollar_per_price_unit" not in meta:
        raise ValueError("_meta.instrument_dollar_per_price_unit missing")

    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if not isinstance(v, dict):
            raise ValueError(f"account '{k}' must be a JSON object")
        for f in REQUIRED_FIELDS:
            if f not in v:
                raise ValueError(f"account '{k}' missing required field '{f}'")
        if v["drawdown_type"] not in DRAWDOWN_TYPES:
            raise ValueError(
                f"account '{k}': drawdown_type='{v['drawdown_type']}' "
                f"not in {sorted(DRAWDOWN_TYPES)}")
        # Nullable fields: daily_loss_limit (firm may not have one),
        # max_challenge_days (untimed evaluations are allowed).
        # Anything else with `null` is rejected.
        for required_numeric in ("starting_balance", "profit_target",
                                  "max_loss", "trailing_drawdown",
                                  "max_contracts", "minimum_trading_days",
                                  "consistency_rule_percent",
                                  "payout_target", "payout_min_days"):
            if v[required_numeric] is None:
                raise ValueError(
                    f"account '{k}': required field '{required_numeric}' "
                    "is null; only daily_loss_limit and max_challenge_days "
                    "may be null")
        # _verification block is optional in schema but the runner gates
        # results research_only without it.
        ver = v.get("_verification", {})
        if ver and not isinstance(ver, dict):
            raise ValueError(f"account '{k}': _verification must be an object")


def load_all() -> dict[str, AccountSpec]:
    raw = json.loads(CONFIG.read_text())
    validate_schema(raw)
    out: dict[str, AccountSpec] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        ver = v.get("_verification", {}) or {}
        out[k] = AccountSpec(
            name=k,
            firm=v["firm"],
            starting_balance=float(v["starting_balance"]),
            profit_target=float(v["profit_target"]),
            daily_loss_limit=(float(v["daily_loss_limit"])
                              if v["daily_loss_limit"] is not None else None),
            max_loss=float(v["max_loss"]),
            trailing_drawdown=float(v["trailing_drawdown"]),
            drawdown_type=v["drawdown_type"],
            max_contracts=int(v["max_contracts"]),
            minimum_trading_days=int(v["minimum_trading_days"]),
            consistency_rule_percent=float(v["consistency_rule_percent"]),
            payout_target=float(v["payout_target"]),
            payout_min_days=int(v["payout_min_days"]),
            max_challenge_days=(int(v["max_challenge_days"])
                                if v["max_challenge_days"] is not None else None),
            source_url=ver.get("source_url"),
            last_verified=ver.get("last_verified"),
            verification_notes=ver.get("notes"),
            extras={kk: vv for kk, vv in v.items()
                    if kk not in REQUIRED_FIELDS and not kk.startswith("_")},
        )
    return out


def verification_status(spec: AccountSpec,
                         today: _dt.date | None = None,
                         stale_days: int = VERIFICATION_STALE_DAYS) -> str:
    """Return verification status for a single account."""
    if (spec.source_url or "").lower() == "synthetic":
        return "synthetic"
    if not spec.last_verified:
        return "unverified"
    try:
        verified_dt = _dt.date.fromisoformat(spec.last_verified)
    except (TypeError, ValueError):
        return "unverified"
    today = today or _dt.date.today()
    age = (today - verified_dt).days
    return "verified" if age <= stale_days else "stale"


def can_certify_for_live(spec: AccountSpec, today: _dt.date | None = None) -> bool:
    """Strict gate: only `verified` accounts may certify a strategy as
    fit for live deployment. `synthetic` is for the harness tests."""
    return verification_status(spec, today) == "verified"


def dollar_per_price_unit(instrument: str = "MGC") -> float:
    raw = json.loads(CONFIG.read_text())
    return float(raw["_meta"]["instrument_dollar_per_price_unit"][instrument])
