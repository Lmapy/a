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

## Stage 3 — agentic search with walk-forward gating

The single-spec hypothesis fails. So we built a search loop that proposes
many specs and gates each through walk-forward and a holdout. See
[`agents/06-proposer.md`](agents/06-proposer.md),
[`agents/07-walkforward.md`](agents/07-walkforward.md),
[`agents/08-critic.md`](agents/08-critic.md), and
[`agents/09-orchestrator.md`](agents/09-orchestrator.md).

```
proposer → walk-forward (~25 disjoint folds, 2018-2026) → holdout (matched 2026 M15) → critic → leaderboard
```

The proposer is currently a deterministic grid over filter / entry / stop
combinations (~648 specs); it is intentionally a placeholder for an LLM
proposer that reads the leaderboard and adapts its next batch.

**Certification rule** (`scripts/orchestrate.py:certify`):

```
walk-forward folds        ≥ 10
walk-forward median Sharpe > 0
walk-forward % positive folds ≥ 0.55
holdout trades             ≥ 30
holdout total return       > 0
```

### Search result

648 specs evaluated in ~2 minutes; **9 certified**. The pattern that
survived: a 50-bar moving-average regime filter (`with`-trend) plus a
body-size or streak filter on the previous bar.

| id                              | wf folds | wf median Sharpe | wf % pos folds | ho trades | ho return | ho Sharpe | certified |
| ------------------------------- | -------: | ---------------: | -------------: | --------: | --------: | --------: | :-------: |
| `body0.5_reg50wit_m15_atr_stop` | 27       | **1.04**         | 0.59           | 52        | +6.11%    | 6.84      | ✅ |
| `body0.5_reg50wit_m15_open`     | 27       | 1.04             | 0.59           | 52        | +3.29%    | 2.64      | ✅ |
| `reg50wit_streak2_m15_open_h4_atrx1` | 27 | 0.41             | 0.56           | 78        | +3.99%    | 2.52      | ✅ |
| `body1.0_reg50wit_m15_open_h4_atrx1` | 27 | **3.12**         | 0.56           | 14        | +2.37%    | 8.58      | ❌ holdout < 30 trades |
| `body1.0_reg50wit_streak2_m15_atr_stop` | 27 | 2.89        | 0.67           | 6         | +2.71%    | 19.9      | ❌ holdout < 30 trades |

The two `body1.0` rows have higher walk-forward Sharpe but get rejected
because the strict body filter leaves too few holdout trades to certify.
That's the critic working: a 27-fold edge isn't enough to call it a
winner if the OOS sample is < 30 trades.

**Important caveats.** The `wf_median_sharpe ≈ 1` certified rows have
average per-fold returns of ~0.4% — small enough that a real cost model
worse than 1.5 bps could erase the edge. The huge holdout Sharpe numbers
(2.6 → 6.8) are annualised from 52-78 trades over 9 weeks; treat those
as *consistency signals*, not as forecasts. The walk-forward median is
the metric to anchor on.

### Conclusion

The user's original "every H4 candle continues the previous one"
hypothesis fails on its own. With one extra ingredient — a 50-bar
**with-trend** regime filter — the continuation idea passes a strict
walk-forward + holdout gate, modestly. The cleanest certified spec is
`body0.5_reg50wit_m15_atr_stop`: take the continuation only when the
prior H4 body is ≥ 0.5 × ATR(14) **and** the prior close is on the
same side of the 50-bar MA as the trade direction.

Future work for the agent loop:

- Replace the grid proposer with an adaptive Claude-API proposer that
  reads the leaderboard.
- Tighten the critic: deflated Sharpe, regime stability across decade
  slices, sensitivity to `cost_bps`.
- Get a longer M15 history so the holdout sample size isn't binding.

## Reproduce

```bash
pip install pandas numpy matplotlib
make data         # fetch_data.py
make backtest     # backtest.py — single-spec headline numbers
make search       # orchestrate.py — agentic search + leaderboard
```

All CSVs and PNG land in `results/`. The strategy spec schema lives in
[`agents/06-proposer.md`](agents/06-proposer.md); the backtester
contract in [`agents/03-backtester.md`](agents/03-backtester.md).
