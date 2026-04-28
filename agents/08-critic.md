# Agent 08 — critic

## Role

Decide whether a candidate strategy is *certified* (worth elevating to
the leaderboard's "winners" view) or rejected. The critic is intentionally
strict: a strategy must look good in two independent places — the
walk-forward folds and the matched M15 holdout — before it counts.

## Inputs

- Walk-forward result (from agent 07).
- Holdout result: `run_full_sim(spec, h4_matched, m15_matched)` →
  `trades_to_metrics(...)`. This is the matched 2026 window with real
  broker spreads.

## Certification rule (`scripts/orchestrate.py:certify`)

```python
def certify(wf, ho):
    return (
        wf["folds"] >= 10                       # enough fold evidence
        and wf["median_sharpe"] > 0
        and wf["pct_positive_folds"] >= 0.55
        and ho["trades"] >= 30                  # enough holdout trades
        and ho["total_return"] > 0
    )
```

If you want a stricter critic later, candidates to add:

- `wf["worst_fold_dd"] >= -0.20` — no single fold imploded.
- `ho["sharpe_ann"] > 0.5` — holdout Sharpe is meaningfully positive.
- Deflated Sharpe vs. the number of strategies tried in this run
  (Bailey & López de Prado 2014).

## Anti-overfit posture

The critic sees the holdout result, **but the holdout is only used as a
gating signal — never as a search objective**. The proposer must not
optimise against the holdout. Today this is enforced structurally
because the proposer is a static grid; if/when an LLM proposer is
introduced, the leaderboard given to the proposer should hide the
holdout columns and only show walk-forward metrics.

## Output

A boolean column `certified` on each leaderboard row. The leaderboard is
sorted with certified rows on top, then by `wf_median_sharpe`,
then by `ho_total_return`.
