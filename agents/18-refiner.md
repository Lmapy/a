# Agent 18 — refiner (agent 06 in v3 spec)

## Role

Take strategies that **failed** the critic or certifier in a *promising*
way (one or two soft fails, not many hard fails) and propose a small
controlled modification. Re-run agent 04 → 05 → 07 on the modified
spec; promote only if the new variant passes more gates and the
original failure mode is gone.

The refiner is the most overfit-prone agent in the workflow and is
heavily fenced.

## Allowed modifications (at most ONE per refinement round)

- tighten or relax the `session` filter
- swap `entry` model among the compatible-by-timeframe set
- swap `stop` from structural (prev_h4_open / prev_h4_extreme) to
  ATR-based (h4_atr ×mult) or vice-versa
- add or remove **one** filter
- change `fib_limit_entry.level` between `{0.382, 0.5, 0.618, 0.705, 0.786}`
- change `body_atr.min` by ≤0.2 from current
- change `regime.ma_n` by ≤30 from current
- change `risk.per_trade_pct`

## Forbidden

- adding more than one filter at a time
- modifying based on holdout-window outcomes
- changing the strategy after seeing the FDR-corrected p-value
  (looking at the holdout once is fine; iterating against it is leakage)
- using future data
- ignoring failure reasons that DO point at a real bug rather than
  a noise miss

## Output

`results/refinement_log.json`:

```json
{
  "round": 1,
  "starting_spec_id": "...",
  "modification": "filters[0].min: 0.5 -> 0.7",
  "result_id": "...",
  "before": { passes_critic: true, certified: false, ho_total_return: 0.054 },
  "after":  { passes_critic: true, certified: false, ho_total_return: 0.062 },
  "decision": "no improvement on cert criteria; revert"
}
```

## Status

The refiner is implemented as a markdown spec only in this commit. It
requires at least one critic-passing spec to refine; today there are
three (all `strong_body_continuation` variants), all failing the
shuffled-outcome and random-baseline tests because of the short M15
holdout. Refining cannot fix a sample-size issue, so refinement is
deliberately deferred until more M15 history is available.
