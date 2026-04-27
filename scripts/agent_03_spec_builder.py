"""Agent 03 — Strategy Spec Builder.

Reads results/hypotheses.json and expands each family template into
concrete specs. Each spec is checked against the entry-model
compatibility map and against what's actually implemented in the
executor; specs that cannot run are emitted with a status field of
"skipped" plus a reason instead of silently dropped.

Output: results/generated_specs.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.strategy_families import STRATEGY_FAMILIES
from entry_models import registry
from entry_models.compatibility import compatibility_status

HYPOTHESES = ROOT / "results" / "hypotheses.json"
OUT = ROOT / "results" / "generated_specs.json"

# What signal/filter/exit types the executor knows how to run today.
KNOWN_SIGNAL_TYPES = {"prev_color", "prev_color_inverse"}
KNOWN_FILTER_TYPES = {
    "body_atr", "session", "regime", "min_streak", "atr_percentile",
    "vwap_dist", "wick_ratio",
}
KNOWN_EXIT_TYPES = {"h4_close", "prev_h4_extreme_tp"}
KNOWN_STOP_TYPES = {"none", "prev_h4_open", "prev_h4_extreme",
                    "h4_atr", "m15_atr"}

# Entry timeframes we have actual bar data for (drives data_unavailable).
AVAILABLE_ENTRY_TIMEFRAMES = {"M15"}


def _set_path(spec: dict, dotted: str, value: Any) -> None:
    """Apply a `filters[0].min` / `entry.level` style override to a spec."""
    parts = dotted.replace("]", "").split(".")
    cur: Any = spec
    for p in parts[:-1]:
        if "[" in p:
            key, idx = p.split("[")
            cur = cur[key][int(idx)]
        else:
            cur = cur[p]
    last = parts[-1]
    if "[" in last:
        key, idx = last.split("[")
        cur[key][int(idx)] = value
    else:
        cur[last] = value


def _spec_known(spec: dict) -> tuple[bool, list[str]]:
    """Returns (is_runnable, list_of_unknown_pieces)."""
    issues = []
    if spec["signal"]["type"] not in KNOWN_SIGNAL_TYPES:
        issues.append(f"unknown signal: {spec['signal']['type']}")
    for f in spec.get("filters", []):
        if f["type"] not in KNOWN_FILTER_TYPES:
            issues.append(f"unknown filter: {f['type']}")
    if spec["entry"]["type"] not in registry.names():
        issues.append(f"unknown entry: {spec['entry']['type']}")
    if spec["stop"]["type"] not in KNOWN_STOP_TYPES:
        issues.append(f"unknown stop: {spec['stop']['type']}")
    if spec["exit"]["type"] not in KNOWN_EXIT_TYPES:
        issues.append(f"unknown exit: {spec['exit']['type']}")
    return (not issues, issues)


def _build_one(family_id: str, family: dict, variant: dict,
               entry_tf: str) -> dict:
    spec = copy.deepcopy(family["template"])
    spec["bias_timeframe"] = "H4"
    spec["setup_timeframe"] = "H4"
    spec["entry_timeframe"] = entry_tf

    # Apply variant overrides. Variants can either fully replace a key
    # (e.g. {"filters": [...]}) or do a dotted-path override.
    for key, val in variant.items():
        if "." in key or "[" in key:
            _set_path(spec, key, val)
        else:
            spec[key] = val

    # Compose deterministic id from family + variant + tf.
    summary_bits = [family_id, entry_tf]
    for k, v in variant.items():
        summary_bits.append(f"{k}={v}")
    spec["id"] = "_".join(summary_bits).replace(" ", "")

    # Compatibility + runnability check.
    em = spec["entry"]["type"]
    data_avail = entry_tf in AVAILABLE_ENTRY_TIMEFRAMES
    cstat = compatibility_status(em, entry_tf, data_avail)

    runnable, issues = _spec_known(spec)
    status = "runnable"
    skip_reasons: list[str] = []
    if cstat == "data_unavailable":
        status = "skipped"
        skip_reasons.append(f"no {entry_tf} bars on disk yet")
    elif cstat == "resolution_limited":
        status = "skipped"
        skip_reasons.append(f"{em} not designed for {entry_tf}")
    elif cstat == "unknown_model":
        status = "skipped"
        skip_reasons.append(f"unknown entry model: {em}")
    if not runnable:
        status = "skipped"
        skip_reasons.extend(issues)

    return {
        "spec": spec,
        "family_id": family_id,
        "entry_timeframe": entry_tf,
        "compatibility_status": cstat,
        "status": status,
        "skip_reasons": skip_reasons,
    }


def run() -> None:
    if not HYPOTHESES.exists():
        raise SystemExit(f"missing {HYPOTHESES}; run agent_02 first")

    out_specs: list[dict] = []
    for family_id, fam in STRATEGY_FAMILIES.items():
        for tf in fam["preferred_entry_timeframes"]:
            for variant in fam.get("variants", [{}]):
                out_specs.append(_build_one(family_id, fam, variant, tf))

    n_run = sum(1 for s in out_specs if s["status"] == "runnable")
    n_skip = len(out_specs) - n_run
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "agent_03_spec_builder",
        "n_total": len(out_specs),
        "n_runnable": n_run,
        "n_skipped": n_skip,
        "available_entry_timeframes": sorted(AVAILABLE_ENTRY_TIMEFRAMES),
        "specs": out_specs,
    }, indent=2))
    print(f"wrote {OUT}  total={len(out_specs)}  runnable={n_run}  "
          f"skipped={n_skip}")


if __name__ == "__main__":
    run()
