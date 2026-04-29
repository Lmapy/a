"""Risk-model sweep.

Take a base PropCandidate; emit a variant per risk-model preset.
Risk presets follow the user's brief:

  fixed_contracts      always trade `contracts_base`
  fixed_dollar_risk    target `dollar_risk_per_trade` per trade
  pct_dd_buffer        risk X% of remaining drawdown buffer
  reduce_after_loss    halve size after the most recent loss
  scale_after_high     +1 size after each new equity high

ATR / stop-distance sizing uses the same RiskBlock — the
prop_challenge.risk.RiskModel.size() function consumes
`stop_distance_price` (or `atr_pre_entry`) at simulation time, so
we don't need a separate "atr_normalised" preset; any preset with
`dollar_risk_per_trade` set will use stop distance internally.

The sweep is intentionally small (5 presets x 1 contracts_max).
The orchestrator can request a wider sweep by composing
`variants_with_caps(base, contracts_caps=(1, 2, 5))`.
"""
from __future__ import annotations

from typing import Iterable

from core.candidate import PropCandidate, RiskBlock


# Each preset is (label, RiskBlock-overrides).
DEFAULT_PRESETS: tuple[tuple[str, dict], ...] = (
    ("fixed_micro_1",
     {"name": "fixed_micro_1", "contracts_base": 1}),
    ("fixed_micro_2",
     {"name": "fixed_micro_2", "contracts_base": 2}),
    ("dollar_risk_50",
     {"name": "dollar_risk_50", "dollar_risk_per_trade": 50.0}),
    ("dollar_risk_100",
     {"name": "dollar_risk_100", "dollar_risk_per_trade": 100.0}),
    ("pct_dd_buffer_2pct",
     {"name": "pct_dd_buffer_2pct", "pct_dd_buffer": 2.0}),
    ("reduce_after_loss",
     {"name": "reduce_after_loss", "contracts_base": 2,
       "reduce_after_loss": True}),
    ("scale_after_high",
     {"name": "scale_after_high", "contracts_base": 1,
       "scale_after_high": True}),
)


def _risk_with(overrides: dict, *, contracts_max: int) -> RiskBlock:
    """Build a RiskBlock from base defaults + overrides."""
    fields = {f: getattr(RiskBlock(), f)
              for f in RiskBlock.__dataclass_fields__}
    fields.update(overrides)
    fields["contracts_max"] = contracts_max
    return RiskBlock(**fields)


def _clone(base: PropCandidate, *,
           risk: RiskBlock, label: str) -> PropCandidate:
    return PropCandidate(
        id=f"{base.id}__risk_{label}",
        family=base.family,
        symbol=base.symbol,
        bias_timeframe=base.bias_timeframe,
        setup_timeframe=base.setup_timeframe,
        entry_timeframe=base.entry_timeframe,
        signal=dict(base.signal),
        filters=[dict(f) for f in base.filters],
        entry=dict(base.entry),
        stop=dict(base.stop),
        exit=dict(base.exit),
        risk=risk,
        daily_rules=base.daily_rules,
        account=base.account,
        cost_model=base.cost_model,
        cost_bps=base.cost_bps,
        seed=base.seed,
        notes=base.notes,
        provenance=dict(base.provenance, lab="risk_sweep",
                         base_risk=base.risk.name),
    )


def variants(base: PropCandidate,
             presets: Iterable[tuple[str, dict]] | None = None,
             *,
             contracts_max: int = 5) -> list[PropCandidate]:
    """Emit one variant per risk preset. `contracts_max` should
    match the prop account's `max_contracts` to avoid sizing past the
    cap; the orchestrator typically sets it from the AccountSpec."""
    presets = list(presets) if presets is not None else list(DEFAULT_PRESETS)
    out: list[PropCandidate] = []
    for label, ov in presets:
        out.append(_clone(base, risk=_risk_with(ov, contracts_max=contracts_max),
                           label=label))
    return out


def variants_with_caps(base: PropCandidate,
                        contracts_caps: tuple[int, ...] = (1, 3, 5),
                        ) -> list[PropCandidate]:
    """Sweep both presets AND contracts_max. Cartesian product is
    small by default (5 presets x 3 caps = 15 variants per base)."""
    out: list[PropCandidate] = []
    for cap in contracts_caps:
        out.extend(variants(base, contracts_max=cap))
    return out


def variants_for_many(bases: Iterable[PropCandidate],
                       presets: Iterable[tuple[str, dict]] | None = None,
                       *,
                       contracts_max: int = 5) -> list[PropCandidate]:
    out: list[PropCandidate] = []
    for b in bases:
        out.extend(variants(b, presets, contracts_max=contracts_max))
    return out
