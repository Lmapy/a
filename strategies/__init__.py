"""Strategy generators + sweep labs (Batch G).

This package emits `PropCandidate` objects for the prop-firm passing
engine. It does NOT run candidates — the orchestrator (Batch H,
`scripts/run_prop_passing.py`) consumes them and produces the
leaderboard.

Modules
-------
  families       deterministic per-family candidate generators
  entry_lab      generate entry-model variants for a base candidate
  risk_sweep     generate risk-model variants for a base candidate
  daily_rule_lab generate daily-rule variants for a base candidate
  grid           controlled tier-1 grid: ALL families x small sweep

Design rules
------------
* Every generator returns a `list[PropCandidate]`.
* Generators are deterministic (no RNG). The orchestrator chooses
  which permutations to run.
* Generators do not mutate their inputs; they return fresh candidates.
* Generators MUST be OHLC-only — every emitted candidate must pass
  `core.feature_capability.classify_candidate(...).status == "ok"`.
"""
