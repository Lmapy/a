# Agent 26 — Drawdown Survival Critic

## Role

Stress-test every (strategy × account × risk × rules) combination
against the realistic ways it can lose:

- losing-streak clustering (block-bootstrap with non-trivial block size)
- worst-case trade ordering (sort losses to fire early)
- spread widening / slippage spikes (already tested by execution
  stress in agent 04 + critic — surfaced here per-account)
- intraday-DD sensitivity (re-run with `intraday_trailing` accounts)
- EOD-DD sensitivity (re-run with `eod_trailing` accounts)

## Implementation

The challenge MC is itself the stress test:

1. `prop_challenge.challenge.run_challenge` does block-bootstrap with
   block=5 trades, so realistic clusters of bad days are sampled.
2. The runner already iterates over Topstep (EOD trailing), MFFU
   (intraday trailing), and Generic-static accounts — that's the
   intraday vs EOD sensitivity test.
3. Per-combo `breach_reason_histogram` is aggregated into
   `results/prop_failure_modes.json`.

## Hard rejection criteria

A strategy/account combo is rejected by this critic if:

- `blowup_probability > 0.50` on any account
- `consistency_breach_rate > 0.30` on any passing path
- single dominant breach reason accounts for `> 80%` of all blowups
  (signals a single fragile boundary)
