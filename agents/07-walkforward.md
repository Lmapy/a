# Agent 07 — walk-forward harness

## Role

Score a strategy spec on the long H4 history in a way that is robust to
single-window luck. The harness returns *aggregate* stability metrics,
not a single backtest number.

## Inputs

- A strategy spec (from agent 06).
- `data/XAUUSD_H4_long.csv` (8.6k bars, 2018-06-28 → 2026-04-20).

## Method

Rolling, non-overlapping test windows:

```
fold k  ─ train ─────────► ─ test ────►
fold k+1            ─ train ─────────► ─ test ────►
fold k+2                       ─ train ─────────► ─ test ────►
```

Defaults: 12-month train, 3-month test, 3-month step → ~25 test folds
that together tile the history end-to-end with no overlap.

The harness does not currently *fit* anything to the train slice (specs
arrive fully parameterised). The train slice exists for two reasons:

1. To provide warm-up for rolling indicators (ATR, MA), so signals at the
   start of the test window aren't truncated.
2. To give a future "calibrating proposer" a place to look when deciding,
   say, what body-ATR threshold to use for the next test slice.

Within each fold, the spec is run via `run_h4_sim` (whole-bar simulation:
enter at H4 open, exit at H4 close, optional H4-ATR stop, fixed
`cost_bps` for round-trip costs). Trades whose `entry_time` falls outside
`[test_start, test_end)` are dropped.

## Outputs (aggregate over folds)

| field                  | meaning                                       |
| ---------------------- | --------------------------------------------- |
| `folds`                | number of fold tests that produced trades     |
| `median_sharpe`        | median annualised Sharpe across folds         |
| `pct_positive_folds`   | share of folds with positive total return     |
| `avg_total_return`     | mean fold total return                        |
| `median_total_return`  | median fold total return                      |
| `worst_fold_dd`        | worst per-fold max drawdown                   |
| `fold_table`           | per-fold breakdown (written to `search_folds.csv`) |

## Why this kills most overfit specs

Picking the strategy with the highest Sharpe in a single window is a
distributional fluke when you try hundreds of variants. Demanding that
the *median* Sharpe across ~25 disjoint windows is positive **and** that
**≥55%** of folds are positive is a much harder bar — it requires the
edge to survive multiple gold regimes (2018 grind-up, 2020 COVID spike,
2022 rate-shock, 2024-26 rally).

## Acceptance checks

- Folds are non-overlapping (each H4 bar appears in at most one test
  window).
- The earliest fold has `train_start` ≥ data start.
- Every fold has ≥ 30 H4 bars in its test window (else: dropped).
