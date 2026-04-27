"""Agent 02 — Hypothesis Generator.

Emits one hypothesis JSON per market-structure principle in
core.strategy_families.STRATEGY_FAMILIES. The output is intentionally
NOT a sweep over indicator combinations -- each hypothesis comes from
named market behaviour with an explicit failure mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.strategy_families import all_hypotheses

OUT = ROOT / "results" / "hypotheses.json"


def run() -> None:
    hyps = list(all_hypotheses())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "agent_02_hypothesis_generator",
        "n_hypotheses": len(hyps),
        "hypotheses": hyps,
    }, indent=2))
    print(f"wrote {OUT}  hypotheses={len(hyps)}")


if __name__ == "__main__":
    run()
