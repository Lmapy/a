"""Daily-rule optimiser.

Take a base PropCandidate; emit a variant per daily-rule preset.
The presets implement the user's brief:

  max_trades_per_day        1, 2, 3, 4
  stop_after_n_losses       1, 2, 3
  stop_after_loss_dollar    -250, -500, -1000
  stop_after_profit_dollar  +250, +500, +1000
  session_only              ny, london, ny_open, london_open
  reduce_after_loss         applies via the RiskBlock; not here
  protect_profit_after_buf  daily profit stop with smaller value

Cooldown ("don't trade for N bars after a loss") is not yet
expressed in `prop_challenge.lockout.DailyRules`. It will be added
in a follow-up; the preset names that imply cooldown are stubbed
with a `metadata.requires_lockout_extension="cooldown"` flag the
orchestrator can detect.

The default sweep is intentionally small to keep candidate counts
controlled. Wider sweeps go through `variants_full(...)`.
"""
from __future__ import annotations

from typing import Iterable

from core.candidate import DailyRulesBlock, PropCandidate


# Compact default preset list — covers the most-impactful axes
# without combinatorial explosion. Each preset is (label, kwargs).
DEFAULT_PRESETS: tuple[tuple[str, dict], ...] = (
    ("none", {}),
    ("max1", {"max_trades_per_day": 1}),
    ("max2", {"max_trades_per_day": 2}),
    ("max3", {"max_trades_per_day": 3}),
    ("stop_l1", {"stop_after_n_losses": 1}),
    ("stop_l2", {"stop_after_n_losses": 2}),
    ("dl250", {"daily_loss_stop_dollar": 250.0}),
    ("dl500", {"daily_loss_stop_dollar": 500.0}),
    ("dp500", {"daily_profit_stop_dollar": 500.0}),
    ("dp500_dl300", {"daily_profit_stop_dollar": 500.0,
                      "daily_loss_stop_dollar": 300.0}),
    ("ny_only_max2", {"session_only": "ny", "max_trades_per_day": 2}),
    ("london_only_max2", {"session_only": "london",
                            "max_trades_per_day": 2}),
)


# Wider sweep when the orchestrator runs the full optimiser. ~25
# presets — still controlled.
WIDE_PRESETS: tuple[tuple[str, dict], ...] = DEFAULT_PRESETS + (
    ("max1_stop_l1", {"max_trades_per_day": 1, "stop_after_n_losses": 1}),
    ("max2_stop_l1", {"max_trades_per_day": 2, "stop_after_n_losses": 1}),
    ("max2_stop_l2", {"max_trades_per_day": 2, "stop_after_n_losses": 2}),
    ("max3_stop_l1", {"max_trades_per_day": 3, "stop_after_n_losses": 1}),
    ("max3_stop_l2", {"max_trades_per_day": 3, "stop_after_n_losses": 2}),
    ("dp250", {"daily_profit_stop_dollar": 250.0}),
    ("dp1000", {"daily_profit_stop_dollar": 1000.0}),
    ("dl1000", {"daily_loss_stop_dollar": 1000.0}),
    ("ny_open_max1", {"session_only": "ny_open", "max_trades_per_day": 1}),
    ("london_open_max1", {"session_only": "london_open",
                            "max_trades_per_day": 1}),
    ("ny_only_dl500", {"session_only": "ny", "daily_loss_stop_dollar": 500.0}),
    ("london_only_dl500", {"session_only": "london",
                            "daily_loss_stop_dollar": 500.0}),
)


def _rules(label: str, kwargs: dict) -> DailyRulesBlock:
    fields = {f: getattr(DailyRulesBlock(), f)
              for f in DailyRulesBlock.__dataclass_fields__}
    fields.update(kwargs)
    fields["name"] = label
    return DailyRulesBlock(**fields)


def _clone(base: PropCandidate, *,
           daily: DailyRulesBlock, label: str) -> PropCandidate:
    return PropCandidate(
        id=f"{base.id}__rules_{label}",
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
        risk=base.risk,
        daily_rules=daily,
        account=base.account,
        cost_model=base.cost_model,
        cost_bps=base.cost_bps,
        seed=base.seed,
        notes=base.notes,
        provenance=dict(base.provenance, lab="daily_rule_lab",
                         base_rules=base.daily_rules.name),
    )


def variants(base: PropCandidate,
             presets: Iterable[tuple[str, dict]] | None = None,
             ) -> list[PropCandidate]:
    """Emit one variant per daily-rule preset (default: small set)."""
    presets = list(presets) if presets is not None else list(DEFAULT_PRESETS)
    out: list[PropCandidate] = []
    for label, kw in presets:
        out.append(_clone(base, daily=_rules(label, kw), label=label))
    return out


def variants_full(base: PropCandidate) -> list[PropCandidate]:
    """The wider preset sweep (~25). Use for the daily-rule optimiser
    stage in the orchestrator's full tier."""
    return variants(base, presets=WIDE_PRESETS)


def variants_for_many(bases: Iterable[PropCandidate],
                       presets: Iterable[tuple[str, dict]] | None = None,
                       ) -> list[PropCandidate]:
    out: list[PropCandidate] = []
    for b in bases:
        out.extend(variants(b, presets))
    return out
