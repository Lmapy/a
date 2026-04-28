"""PropCandidate — the rich research object.

A `PropCandidate` is the unit the prop-firm passing engine ranks. It
combines:

  * the strategy (signal + filters + entry + stop + exit)
  * the risk model
  * the daily rules
  * the prop account
  * the cost model
  * provenance + certification metadata

Composition (NOT inheritance) over the existing `core.types.Spec`:

  * `candidate.to_spec()` -> a `Spec` the hardened executor / walk-
    forward / holdout consume unchanged.
  * `PropCandidate.from_spec(spec, ...)` -> wrap an existing Spec
    with default risk / daily / account / certification fields.

This protects the hardened executor path. Anything already running
through `Spec` keeps working; the new prop-passing engine consumes
the richer `PropCandidate` and exports a Spec when it needs to call
the executor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from core.certification import (
    CertificationLevel, CertificationVerdict, FailureReason,
    rank_level, verdict_from_reasons,
)
from core.types import Spec


# --- side blocks -------------------------------------------------------------

@dataclass
class RiskBlock:
    """Pre-trade-only risk-sizing parameters for a candidate.

    Maps onto the prop_challenge.risk.RiskModel constructor. Stored
    here as a plain dict so candidates serialise cleanly to JSON.
    """
    name: str = "fixed_contracts_1"
    contracts_base: int = 1
    contracts_max: int = 5
    dollar_risk_per_trade: float | None = None
    pct_dd_buffer: float | None = None
    reduce_after_loss: bool = False
    scale_after_high: bool = False

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict | None) -> "RiskBlock":
        if not d:
            return cls()
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class DailyRulesBlock:
    """Daily-rules / lockout config. Maps onto prop_challenge.lockout.DailyRules."""
    name: str = "none"
    max_trades_per_day: int | None = None
    stop_after_n_wins: int | None = None
    stop_after_n_losses: int | None = None
    daily_profit_stop_dollar: float | None = None
    daily_loss_stop_dollar: float | None = None
    session_only: str | None = None     # asia | london | ny | london_open | ny_open

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict | None) -> "DailyRulesBlock":
        if not d:
            return cls()
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class AccountRef:
    """Reference to one of the verified prop accounts in
    config/prop_accounts.json. Looked up by name at simulation time."""
    name: str = "topstep_50k"
    instrument: str = "MGC"
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict | None) -> "AccountRef":
        if not d:
            return cls()
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


# --- the candidate -----------------------------------------------------------

@dataclass
class PropCandidate:
    """Full prop-passing research object."""
    # identity
    id: str
    family: str = "unspecified"
    symbol: str = "XAUUSD"

    # timeframes (mirror Spec)
    bias_timeframe: str = "H4"
    setup_timeframe: str = "H4"
    entry_timeframe: str = "M15"

    # strategy logic (mirror Spec, plain dicts so JSON round-trips cleanly)
    signal: dict = field(default_factory=lambda: {"type": "prev_color"})
    filters: list[dict] = field(default_factory=list)
    entry: dict = field(default_factory=lambda: {"type": "touch_entry"})
    stop: dict = field(default_factory=lambda: {"type": "prev_h4_open"})
    exit: dict = field(default_factory=lambda: {"type": "h4_close"})

    # rich blocks
    risk: RiskBlock = field(default_factory=RiskBlock)
    daily_rules: DailyRulesBlock = field(default_factory=DailyRulesBlock)
    account: AccountRef = field(default_factory=AccountRef)

    # cost / execution
    cost_model: str = "spread"          # "spread" | "bps"
    cost_bps: float = 1.5
    seed: int | None = None

    # research metadata
    notes: str = ""
    provenance: dict = field(default_factory=dict)

    # certification + audit trail
    certification_level: CertificationLevel = CertificationLevel.CANDIDATE
    failure_reasons: list[FailureReason] = field(default_factory=list)
    rejection_detail: dict = field(default_factory=dict)

    # ---- adaptors ----------------------------------------------------------

    def to_spec(self) -> Spec:
        """Project to the executor-compatible `Spec`. The hardened
        executor / walk-forward / holdout / statistical tests all
        consume Spec, so this is the boundary that keeps the canonical
        pipeline stable."""
        return Spec(
            id=self.id,
            bias_timeframe=self.bias_timeframe,
            setup_timeframe=self.setup_timeframe,
            entry_timeframe=self.entry_timeframe,
            signal=dict(self.signal),
            filters=[dict(f) for f in self.filters],
            entry=dict(self.entry),
            stop=dict(self.stop),
            exit=dict(self.exit),
            risk={"name": self.risk.name},   # `Spec.risk` is a dict; the
                                             # full RiskBlock lives on the
                                             # candidate, not the Spec
            cost_bps=self.cost_bps,
            seed=self.seed,
        )

    @classmethod
    def from_spec(cls, spec: Spec, *,
                   family: str = "unspecified",
                   risk: RiskBlock | None = None,
                   daily_rules: DailyRulesBlock | None = None,
                   account: AccountRef | None = None,
                   cost_model: str = "spread",
                   notes: str = "",
                   ) -> "PropCandidate":
        """Wrap an existing Spec in a PropCandidate with default
        risk / daily / account if not supplied. Useful for migrating
        legacy specs into the new engine."""
        return cls(
            id=spec.id,
            family=family,
            bias_timeframe=spec.bias_timeframe,
            setup_timeframe=spec.setup_timeframe,
            entry_timeframe=spec.entry_timeframe,
            signal=dict(spec.signal),
            filters=[dict(f) for f in spec.filters],
            entry=dict(spec.entry),
            stop=dict(spec.stop),
            exit=dict(spec.exit),
            risk=risk or RiskBlock(),
            daily_rules=daily_rules or DailyRulesBlock(),
            account=account or AccountRef(),
            cost_model=cost_model,
            cost_bps=spec.cost_bps,
            seed=spec.seed,
            notes=notes,
        )

    # ---- certification helpers --------------------------------------------

    def apply_verdict(self, verdict: CertificationVerdict) -> None:
        self.certification_level = verdict.level
        self.failure_reasons = list(verdict.failure_reasons)
        self.rejection_detail.update(verdict.detail)

    def has_failure(self, reason: FailureReason) -> bool:
        return reason in self.failure_reasons

    # ---- JSON round-trip --------------------------------------------------

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "family": self.family,
            "symbol": self.symbol,
            "bias_timeframe": self.bias_timeframe,
            "setup_timeframe": self.setup_timeframe,
            "entry_timeframe": self.entry_timeframe,
            "signal": self.signal,
            "filters": self.filters,
            "entry": self.entry,
            "stop": self.stop,
            "exit": self.exit,
            "risk": self.risk.to_json(),
            "daily_rules": self.daily_rules.to_json(),
            "account": self.account.to_json(),
            "cost_model": self.cost_model,
            "cost_bps": self.cost_bps,
            "seed": self.seed,
            "notes": self.notes,
            "provenance": self.provenance,
            "certification_level": self.certification_level.value,
            "failure_reasons": [r.value for r in self.failure_reasons],
            "rejection_detail": self.rejection_detail,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "PropCandidate":
        return cls(
            id=payload["id"],
            family=payload.get("family", "unspecified"),
            symbol=payload.get("symbol", "XAUUSD"),
            bias_timeframe=payload.get("bias_timeframe", "H4"),
            setup_timeframe=payload.get("setup_timeframe", "H4"),
            entry_timeframe=payload.get("entry_timeframe", "M15"),
            signal=dict(payload.get("signal", {})),
            filters=[dict(f) for f in payload.get("filters", [])],
            entry=dict(payload.get("entry", {})),
            stop=dict(payload.get("stop", {})),
            exit=dict(payload.get("exit", {})),
            risk=RiskBlock.from_json(payload.get("risk")),
            daily_rules=DailyRulesBlock.from_json(payload.get("daily_rules")),
            account=AccountRef.from_json(payload.get("account")),
            cost_model=payload.get("cost_model", "spread"),
            cost_bps=float(payload.get("cost_bps", 1.5)),
            seed=payload.get("seed"),
            notes=payload.get("notes", ""),
            provenance=dict(payload.get("provenance", {})),
            certification_level=CertificationLevel(
                payload.get("certification_level", CertificationLevel.CANDIDATE.value)),
            failure_reasons=[FailureReason(r)
                             for r in payload.get("failure_reasons", [])],
            rejection_detail=dict(payload.get("rejection_detail", {})),
        )

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), separators=(",", ":"))


# ---- convenience -----------------------------------------------------------

def reject_unavailable_data(candidate: PropCandidate,
                             unavailable_features: list[str],
                             unavailable_tokens: list[str]) -> PropCandidate:
    """Mark a candidate as REJECTED_UNAVAILABLE_DATA in-place.
    Returns the same object for fluent use."""
    candidate.apply_verdict(verdict_from_reasons(
        [FailureReason.REJECTED_UNAVAILABLE_DATA],
        best_possible=CertificationLevel.CANDIDATE,
        detail={
            "unavailable_features": list(unavailable_features),
            "unavailable_tokens": list(unavailable_tokens),
        },
    ))
    return candidate
