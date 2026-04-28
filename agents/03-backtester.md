# Agent 03 — backtester

## Role

Turn the strategy spec (`02-strategy-spec.md`) into a list of trades on
real H4 + M15 data, with no look-ahead.

## Inputs

- `data/XAUUSD_H4_long.csv` — Stage-1 hit-rate diagnostic.
- `data/XAUUSD_H4_matched.csv` — Stage-2 H4 signal source.
- `data/XAUUSD_M15_matched.csv` — Stage-2 M15 entry execution.

## Outputs

- `results/hit_rate.csv` — long-history conditional probabilities.
- `results/trades.csv` — per-trade ledger across all variants:
  `entry_time, exit_time, dir, entry, exit, cost, pnl, ret, strategy`.
- `results/summary.csv` — per-variant aggregate metrics.
- `results/equity.csv`, `results/equity.png` — equity curves.

## Two-stage design

### Stage 1 — hit-rate diagnostic (no trading)

On the long H4 series, compute:

- `P(color_t == color_{t-1})` with a 95% Wald CI.
- Conditional `P(up | prev up)` and `P(down | prev down)`.
- Edge over 50% in percentage points.

This answers **whether the hypothesis is supported by the data at all**.
If the edge is statistically indistinguishable from zero, the executed
backtest is mostly a sanity check.

### Stage 2 — executed backtest with M15 entries

Walk H4 bars in order. For each bar `t` with non-zero `sig_t`:

1. Bucket the M15 series by `floor(time, 4h)` and pull the sub-bars whose
   bucket equals the H4 bar's open time.
2. Pick the entry sub-bar according to the variant (`m15_open`,
   `m15_confirm`, or `m15_atr_stop`). Record `entry_time` and `entry`.
3. If the variant uses a stop (`m15_atr_stop`), iterate sub-bars from entry
   forward and check whether high/low touches the stop. If yes, exit there.
4. Otherwise exit at the H4 bar's close.
5. Subtract round-trip spread as `(spread_entry + spread_exit) * 0.001`.

## Look-ahead audits (must pass)

- The signal at H4 bar `t` only uses `t-1`'s open/close.
- The entry M15 sub-bar's index is ≥ 0 within the H4 bucket of `t` —
  never from a future bucket.
- The stop hit, when it fires, fires on a sub-bar whose `time ≥ entry_time`.
- No metric uses `close_t` for any bar that hasn't completed at decision
  time.

## Implementation

`scripts/backtest.py`. Run it after `fetch_data.py`:

```bash
python3 scripts/backtest.py
```
