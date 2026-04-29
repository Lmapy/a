"""Entry-model lab.

Tests entry models independently from setup logic. Two modes:

  variants(base, entries=...)
      Take one base candidate; emit a copy per entry model. Used to
      ask "is this setup viable only with reclaim_close, or does it
      also work with touch_entry?".

  compare_table(results)
      Given the leaderboard rows for a base setup x its entry
      variants, produce a small comparison table that surfaces the
      best entry per metric. Pure data-shape; the runner produces
      the inputs.

The entry registry lives in `entry_models.registry` (not duplicated
here). `compatibility_status` from `entry_models.compatibility` is
used to skip entries that aren't valid for the candidate's
`entry_timeframe`.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from core.candidate import PropCandidate
from entry_models.compatibility import compatibility_status
from entry_models.registry import names as entry_registry_names


# Default battery of entry models to test against any setup. The
# orchestrator can pass a smaller subset when running fast tiers.
DEFAULT_ENTRIES: tuple[dict, ...] = (
    {"type": "touch_entry"},
    {"type": "reaction_close"},
    {"type": "fib_limit_entry", "level": 0.382},
    {"type": "fib_limit_entry", "level": 0.5},
    {"type": "fib_limit_entry", "level": 0.618},
    {"type": "zone_midpoint_limit"},
    {"type": "minor_structure_break"},
    {"type": "delayed_entry_1"},
    {"type": "delayed_entry_2"},
)


def _entry_id(entry: dict) -> str:
    t = entry.get("type", "?")
    if t == "fib_limit_entry":
        return f"fib{entry.get('level', '?')}"
    return t


def _clone_candidate(base: PropCandidate, *,
                     entry: dict, id_suffix: str) -> PropCandidate:
    """Deep-ish clone of `base` overriding only the entry block + id."""
    return PropCandidate(
        id=f"{base.id}__entry_{id_suffix}",
        family=base.family,
        symbol=base.symbol,
        bias_timeframe=base.bias_timeframe,
        setup_timeframe=base.setup_timeframe,
        entry_timeframe=base.entry_timeframe,
        signal=dict(base.signal),
        filters=[dict(f) for f in base.filters],
        entry=dict(entry),
        stop=dict(base.stop),
        exit=dict(base.exit),
        risk=base.risk,
        daily_rules=base.daily_rules,
        account=base.account,
        cost_model=base.cost_model,
        cost_bps=base.cost_bps,
        seed=base.seed,
        notes=base.notes,
        provenance=dict(base.provenance,
                         lab="entry_model_lab",
                         base_entry_type=base.entry.get("type")),
    )


def variants(base: PropCandidate,
             entries: Iterable[dict] | None = None,
             *,
             skip_incompatible: bool = True) -> list[PropCandidate]:
    """Produce one variant per entry model.

    `skip_incompatible` (default True): if the entry model is not
    compatible with the candidate's `entry_timeframe`, the variant is
    skipped. Set False to emit anyway (the runner can then mark them
    `data_unavailable` itself).
    """
    entries = list(entries) if entries is not None else list(DEFAULT_ENTRIES)
    known = set(entry_registry_names())
    out: list[PropCandidate] = []
    for e in entries:
        et = e.get("type", "")
        if et not in known:
            # not a registered entry model — skip silently
            continue
        if skip_incompatible:
            status = compatibility_status(et, base.entry_timeframe,
                                            data_available=True)
            if status != "ok":
                continue
        out.append(_clone_candidate(base, entry=e, id_suffix=_entry_id(e)))
    return out


def variants_for_many(bases: Iterable[PropCandidate],
                      entries: Iterable[dict] | None = None,
                      ) -> list[PropCandidate]:
    """Apply `variants` to a list of bases; flat output."""
    out: list[PropCandidate] = []
    for b in bases:
        out.extend(variants(b, entries))
    return out


# ---- comparison helper -----------------------------------------------------

def compare_table(results: list[dict],
                   *,
                   metric_keys: tuple[str, ...] = (
                       "ho_total_return",
                       "ho_sharpe_trade_ann",
                       "ho_max_drawdown",
                       "wf_median_sharpe",
                       "label_perm_p",
                       "prop_pass_probability",
                   )) -> dict:
    """Given leaderboard rows for one base setup x its entry variants,
    surface the best entry per metric.

    Each row in `results` must be a dict with at least:
        candidate_id        unique id
        entry_type          the entry model's type
        <metric_keys>       numeric metric values

    Returns:
        {
            "n_entries": <int>,
            "best": {metric -> {"entry": <type>, "value": <float>,
                                  "candidate_id": <id>}},
            "rows": [<input rows>]
        }
    """
    if not results:
        return {"n_entries": 0, "best": {}, "rows": []}
    best: dict[str, dict] = {}
    for k in metric_keys:
        # default sort: higher-is-better; for *_p (p-values) and
        # *_drawdown (negative) flip to lower-is-better.
        higher_is_better = not (k.endswith("_p") or "drawdown" in k)
        valid = [r for r in results if k in r and r[k] is not None]
        if not valid:
            continue
        winner = max(valid, key=lambda r: r[k]) if higher_is_better \
                 else min(valid, key=lambda r: r[k])
        best[k] = {
            "entry": winner.get("entry_type"),
            "candidate_id": winner.get("candidate_id"),
            "value": winner[k],
        }
    return {"n_entries": len(results), "best": best, "rows": list(results)}
