# Agent 23 — Pass Strategy Hunter

## Role

Among all strategies that have a trade ledger in
`results/alpha_trades/`, surface the ones whose **shape** suits a prop
challenge:

- 1–3 trades per day (fits daily-loss budgets)
- high win rate or high profit factor
- losses controlled (low MAE)
- session-specific edge (so a session lockout helps)
- repeatable setup (low variance across folds)

## Method

Read the per-strategy ledger plus the leaderboard row. Score each by:

```
hunter_score = profit_factor * win_rate * trades_in_3to5_per_week_band
             - mae_p90_normalised
```

The top-N by `hunter_score` are passed to agents 22-26 for full prop
simulation. Strategies that fail prop simulation come back via the
critic and refiner; those that survive are passed to the certifier.

## Output

The ranked input list to `scripts/run_prop_challenge.py`. Implemented
inline (not a standalone script) — `collect_strategies()` returns
every ledger and the prop runner iterates accounts × risk × rules over
each.
