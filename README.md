# 4H continuation strategy on gold — backtest pipeline

A reproducible pipeline that tests the user's hypothesis on **real** XAUUSD
data, with **15-minute entries on top of a 4-hour signal**:

> On every 4-hour candle open, gold tends to continue in the same direction
> as the previous 4-hour candle.

No synthetic data is used anywhere in this repo.

## Repo layout

```
agents/        markdown specs for each pipeline stage (00-05)
data/          real OHLC CSVs pulled by scripts/fetch_data.py
scripts/       fetch_data.py, backtest.py
results/       hit_rate.csv, summary.csv, trades.csv, equity.csv, equity.png
```

## Data

All bars come from public MT5 broker exports on GitHub. None of it is
generated.

| File                          | Bars  | Span (UTC)                    | Source                                     |
| ----------------------------- | ----- | ----------------------------- | ------------------------------------------ |
| `data/XAUUSD_H4_long.csv`     | 8,607 | 2018-06-28 → 2026-04-20       | `github.com/142f/inv-cry`                  |
| `data/XAUUSD_H4_matched.csv`  |   261 | 2026-01-30 → 2026-04-01       | `github.com/tiumbj/Bot_Data_Basese`        |
| `data/XAUUSD_M15_matched.csv` | 3,977 | 2026-01-30 → 2026-04-01       | `github.com/tiumbj/Bot_Data_Basese`        |

The matched files are H4 + M15 from the **same broker** so M15 sub-bars
slot cleanly inside H4 buckets.

## Stage 1 — does the prior 4h candle predict the next?

On 8,604 prior/next H4 pairs from 2018→2026 we measure the empirical
probability that the next bar's color equals the previous bar's color:

| Metric                 | Value                  |
| ---------------------- | ---------------------- |
| Bars tested            | 8,604                  |
| P(same direction)      | **0.4971**             |
| 95% Wald CI            | [0.4865, 0.5077]       |
| P(up &#124; prev up)   | 0.5197                 |
| P(down &#124; prev down) | 0.4723               |
| Edge over 50% (pp)     | **−0.29**              |

**Interpretation.** The 95% confidence interval for `P(same)` straddles
0.5, so on the long history the continuation edge is not statistically
distinguishable from a coin flip — actually marginally negative. There is
a mild asymmetry: an up bar is a slight tailwind for the next bar, a down
bar is a slight headwind, but neither is large enough to clear costs.

## Stage 2 — executed backtest with M15 entries

Window: `2026-01-30 → 2026-04-01` (~9 weeks, 261 H4 bars, 3,977 M15 bars).
Costs: round-trip broker spread, full `(spread_entry + spread_exit) *
point_size`. No leverage; each trade is reported as price return on
entry.

| Strategy        | Trades | Win rate | Avg ret (bp) | Total return | Sharpe (ann.) | Max DD   | Avg hold |
| --------------- | -----: | -------: | -----------: | -----------: | ------------: | -------: | -------: |
| `m15_open`      |   260  |  48.85%  |       −2.42  |       −7.40% |        −0.92  |  −18.83% |  214 min |
| `m15_confirm`   |   260  |  47.31%  |       −3.50  |       −9.80% |        −1.42  |  −17.64% |  197 min |
| `m15_atr_stop`  |   260  |  26.54%  |       −0.96  |       −3.12% |        −0.52  |  −14.39% |   99 min |

![equity curves](results/equity.png)

**`m15_open`** — Enter at the open of the first M15 sub-bar inside the new
H4 candle, exit at H4 close. This is the literal version of the
hypothesis. It loses, consistent with Stage 1's near-zero edge being
swamped by spread.

**`m15_confirm`** — Wait for the first M15 candle in the new H4 to confirm
the predicted direction, then enter at its close. This pays to wait, and
the wait costs more than the confirmation buys — the variant is the worst
of the three on this window.

**`m15_atr_stop`** — Same entry as `m15_open` but with a 1×M15-ATR(14)
stop. The stop catches losers earlier (avg hold 99 min vs 214 min) and
reduces total drawdown, but win rate collapses to 27% because in a noisy
9-week window the stop is hit before the H4 plays out.

Skipped-trade diagnostics: 1 skipped (first H4 has no prior signal); 0
skipped for missing M15 sub-bars; 0 skipped in `m15_confirm` for lack of
confirmation. The pipeline is healthy.

## Conclusion

On gold (XAUUSD), the "next 4h candle continues the previous one"
hypothesis is **not supported by the data**. The long-history continuation
probability is 49.7% with a 95% CI that includes 50%, and the executed
backtest with realistic M15 entries and broker spread costs loses money
across all three variants.

This is the cleanest possible negative result: the hypothesis is precisely
stated, the data is real and broad, the execution model uses M15 entries
as requested, and costs are subtracted at the broker level. If a future
variant of the strategy is to be tested, it should add information not
present in just the previous candle's color — for example, body size,
volume confirmation, time-of-day filtering, or higher-timeframe trend.

## Reproduce

```bash
pip install pandas numpy matplotlib
python3 scripts/fetch_data.py
python3 scripts/backtest.py
```

All three CSVs and `equity.png` will land in `results/`. The strategy
rules live in [`agents/02-strategy-spec.md`](agents/02-strategy-spec.md);
update them there first if you change the rules in `scripts/backtest.py`.
