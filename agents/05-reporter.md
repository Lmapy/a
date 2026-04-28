# Agent 05 — reporter

## Role

Write up the result honestly. The reporter never massages numbers; if the
strategy lost money, the report says so. The reporter also makes the
conclusion section actionable: what would meaningfully change the result,
and what would just be over-fitting on this short window?

## Inputs

- `results/hit_rate.csv`
- `results/summary.csv`
- `results/equity.png`
- `agents/02-strategy-spec.md` (rules to quote verbatim)

## Outputs

- `README.md` at the repo root, with these sections in order:
  1. **Hypothesis** — quote the strategy spec verbatim.
  2. **Data** — the three CSVs, where they came from, and the exact spans.
  3. **Stage 1: does the prior bar predict the next bar?** — the long-history
     hit-rate table with its 95% CI; one paragraph of interpretation.
  4. **Stage 2: executed backtest** — the summary table, the equity chart,
     a one-paragraph interpretation per variant, and the diagnostics
     (skipped trades).
  5. **Conclusion** — a short, blunt verdict.
  6. **Reproduce** — the two commands needed.

## Style

- No hype. State what the numbers say.
- If `P(same direction) - 0.5` is inside the 95% CI of zero, write that the
  edge is **not statistically distinguishable from a coin flip**.
- If a variant lost money but a tweak (e.g. adding a body filter) is
  obvious, suggest it as **future work, not as a fix to backfill into the
  conclusion** — the conclusion describes what we just measured.
