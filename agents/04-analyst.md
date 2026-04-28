# Agent 04 — analyst

## Role

Read `results/trades.csv` and `results/hit_rate.csv` and produce the
metrics and plot that go into the report. This agent is intentionally
small — almost everything it computes is also computed inside
`scripts/backtest.py`. It exists as a separate step so the report can be
regenerated from the trade ledger without re-simulating.

## Inputs

- `results/trades.csv`
- `results/hit_rate.csv`

## Outputs

- `results/summary.csv` — one row per strategy variant with:
  - `strategy, trades, wins, win_rate, avg_ret_bp, total_return,
    sharpe_ann, max_drawdown, avg_hold_min`
- `results/equity.csv` — index = exit_time, columns = strategy variant,
  values = `(1 + ret).cumprod()`.
- `results/equity.png` — same data as a line chart.

## Annualisation

H4 gold trades roughly 6 bars per day, 5 days a week → ~1,560 H4 bars/yr.
Sharpe is computed as `mean / std * sqrt(1560)`. This is a per-trade Sharpe
(one observation per H4 bar in the matched window), not a daily-rebalanced
PnL Sharpe, so cross-strategy comparisons are valid but absolute numbers
should be read with that in mind.

## Drawdown

`max_drawdown` is computed on the trade-by-trade equity curve, not on
intra-trade marks. This is conservative: it ignores adverse excursions
inside a trade and reports only the realised drawdown.

## Sanity checks the analyst should reject

- A variant whose `trades` count is 0 (broken signal logic).
- A variant whose `total_return` is exactly 0 across many trades (broken
  PnL accounting, e.g. always exiting at the entry price).
- A variant whose `avg_hold_min` is `> 240` (the H4 bar is 240 minutes;
  anything longer means the exit logic leaked across H4 buckets).
