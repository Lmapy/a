# Agent 11 — fib analyzer

## Role

Diagnostic. Independently of any execution model, measure how price
behaves at standard Fibonacci retracement levels of the previous H4
candle, so we can answer the user's question:

> "the highest level of the previous candle it responds to"

## Method

For each H4 bar `i` whose previous bar has non-zero color, project Fib
levels onto the previous bar's range:

```
long  (after green prev): level_price = prev_high - f * (prev_high - prev_low)
short (after red prev):   level_price = prev_low  + f * (prev_high - prev_low)
```

Levels evaluated: `f ∈ {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0}`.

For each level we record three things on the *current* H4 bar:

| metric                       | definition                                                                  |
| ---------------------------- | --------------------------------------------------------------------------- |
| **touched**                  | did the bar's range reach the level price?                                  |
| **reaction**                 | touched AND the bar closed in the trade direction?                          |
| **ret_from_touch**           | signed return from the level price to H4 close (in trade direction)         |

Aggregations: touch rate, reaction rate, reaction-given-touch, mean and
median ret-from-touch in basis points, win rate from touch. Also a
**deepest-level-touched** distribution per H4 bar, which tells us how
deep retracements typically go.

## Inputs

`data/XAUUSD_H4_long.csv` (8,604 bar pairs over 2018-06-28 → 2026-04-20),
plus the matched 2026 window for cross-check.

## Outputs

- `results/fib_levels.csv` — aggregate per `(dataset, level)`.
- `results/fib_deepest.csv` — distribution of deepest level touched.

## Run

```bash
python3 scripts/fib_analysis.py
make fib    # if added to Makefile
```

## What the long history says

On 8,604 prior/next H4 pairs from 2018–2026:

| level  | touch rate | reaction rate | win rate from touch | mean ret (bp) |
| -----: | ---------: | ------------: | ------------------: | ------------: |
| 0.236  | 0.91       | 0.42          | 0.42                | −7.0          |
| 0.382  | 0.79       | 0.33          | 0.46                | −3.8          |
| 0.500  | 0.68       | 0.26          | 0.48                | −2.6          |
| **0.618** | **0.58** | **0.20**      | **0.50**            | **−1.9**      |
| 0.786  | 0.44       | 0.12          | 0.50                | −1.7          |
| 1.000  | 0.32       | 0.07          | 0.49                | −2.4          |

Deepest level reached in each H4 bar (long history):

| deepest reached | share |
| --------------: | ----: |
| 1.000 (full)    | 32 %  |
| 0.786           | 13 %  |
| 0.618           | 13 %  |
| 0.500           | 11 %  |
| 0.382           | 11 %  |
| 0.236           | 12 %  |
| 0.000 (none)    |  8 %  |

### Reading the table

- **0.618 is the deepest level whose win-rate-from-touch is ≥ 50 %.**
  Below 0.618, you are catching a falling knife (deeper retraces against
  you, worse continuation odds). Above 0.618, win rate stays roughly
  flat at 50 % but touch rate halves and average return-from-touch
  shrinks toward zero.
- **A full retrace happens 32 % of the time.** That's a lot. The
  conventional fib lore that "deep retracement = strong support" does
  not hold up on gold H4 — price simply blows through the previous
  candle's range about a third of the time.
- **Mean return-from-touch is negative at every level.** That tells us
  *naive* fib entries on every signal lose money on average. Selecting
  trades by signal quality (the body+regime+streak filters that
  certified above) is what produces a tradeable edge.

## Why this matters for the strategy

The agentic search now treats fib level as a parameter. Of the 192
fib-retracement specs in the 3–5 trades/week band, **only `fib 0.382`
combinations certified under walk-forward**. That's consistent with this
table: shallower fib levels have higher touch rates, so under tight
filtering the trade frequency stays in the target band; deeper levels
drop frequency below 3/week before the filters even bite.
