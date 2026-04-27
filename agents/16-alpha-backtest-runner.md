# Agent 16 — alpha backtest runner (agent 04 in v3 spec)

## Role

Take every runnable spec from agent 03 and run it through the realistic
executor with full no-lookahead discipline:

- signal at H4 bar `t` only uses bars strictly < `t`
- entry sub-bar must lie inside H4 bucket `t` (verified by validator)
- stops/targets are checked bar by bar; spread on the EXIT leg comes
  from the actual exit bar (post-fix from Phase 1.1)

For each spec it produces:

- the trade ledger (saved to `results/alpha_trades/<spec_id>.csv`)
- walk-forward result (≥20 disjoint folds, 9-mo train / 3-mo test)
- holdout under realistic execution
- holdout under STRESS execution (slippage ×2, spread ×1.5)
- shuffled-outcome p-value, random-baseline p-value
- yearly multi-segment performance
- 25k/50k/150k prop-firm survival simulation
- regime breakdown (session × ret)

## Inputs

- `results/generated_specs.json`
- `data/XAUUSD_*.csv`

## Outputs

Folded into `results/leaderboard.csv` (per-spec row), plus
`results/alpha_trades/<spec_id>.csv` per spec.

## Run

Inside `scripts/run_alpha.py` — not run directly.
