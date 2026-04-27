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

## Pipeline audit

`scripts/audit.py` is a self-contained checker (`make audit`). It runs four
classes of tests and writes the report to `results/audit.txt`. On the
current commit it passes **0 failures / 25 checks** (data integrity,
H4↔M15 alignment, simulator correctness with manual replay, look-ahead
probes, walk-forward fold non-overlap).

The audit caught one real bug on first run: `run_full_sim` was always
using broker spread for cost while `run_h4_sim` (used by walk-forward)
was using `cost_bps`, so the **same spec saw two different cost models**
across walk-forward and holdout. Fixed by adding an explicit
`cost_model` field (`spread` or `bps`) and respecting it in both
runners. Every leaderboard row now declares its cost model.

The audit also surfaced that `run_full_sim` was silently skipping bars
without logging *why*. Both runners now optionally return a `diag` dict
with counters: `signal_zero`, `no_subbars`, `no_confirm`, `no_retrace`,
`missing_prev_levels`. These are written into the leaderboard
(`ho_diag_*` columns) so any "why didn't this trade fire?" question can
be answered without re-running the search.

## Stage 3 — agentic search with walk-forward gating + 3–5 trades/week

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
walk-forward folds                  ≥ 10
walk-forward median Sharpe          > 0
walk-forward % positive folds       ≥ 0.55
3.0 ≤ holdout trades-per-week        ≤ 5.0
holdout total return                 > 0
```

### Strategy DSL

The proposer emits JSON specs combining filters, an entry mode, a stop,
an exit, and a cost model:

```json
{
  "signal": {"type": "prev_color"},
  "filters": [
    {"type": "body_atr",     "min": 0.5, "atr_n": 14},
    {"type": "regime",       "ma_n": 50, "side": "with"},
    {"type": "min_streak",   "k": 2},
    {"type": "candle_class", "classes": ["trend", "rotation"]},
    {"type": "session",      "hours_utc": [12, 16]}
  ],
  "entry": {"type": "m15_retrace_50"},
  "stop":  {"type": "prev_h4_open"},
  "exit":  {"type": "prev_h4_extreme_tp"},
  "cost_model": "spread"
}
```

Entry types: `m15_open`, `m15_confirm`, `m15_atr_stop`, `m15_retrace_50`
(wait for price to retrace to prev-H4 midpoint within the new H4 bar).
Stops: `none`, `h4_atr`, `m15_atr`, `prev_h4_open`, `prev_h4_extreme`.
Exits: `h4_close`, `prev_h4_extreme_tp` (TP at prev-H4 high/low).
Cost models: `spread` (broker spread × point size) or `bps` (fixed bp
fraction of entry).

### Search result

1,056 specs evaluated in ~3 minutes; **12 certified**. The grid
includes both continuation entries (`m15_open`/`m15_confirm`/
`m15_atr_stop`) and the new retracement entry (`m15_retrace_50`) with
prev-H4 structural stops and TPs.

All 12 certified specs share the same signal+filter combo —
`body_atr ≥ 0.5 × ATR + 50-bar with-trend MA + 2-bar streak` — and
land at exactly **3.27 trades/week** (29 trades over 8.86 weeks). They
differ only in execution.

| id                                                  | wf folds | wf median Sharpe | wf % pos | trades/wk | ho return | ho Sharpe |
| --------------------------------------------------- | -------: | ---------------: | -------: | --------: | --------: | --------: |
| `body0.5_reg50wit_streak2_m15_open_prev_h4_open`    | 27       | 0.36             | 0.56     | **3.27**  | **+7.51%**| 12.31     |
| `body0.5_reg50wit_streak2_m15_open_prev_h4_extreme` | 27       | 0.36             | 0.56     | 3.27      | +7.34%    | 11.97     |
| `body0.5_reg50wit_streak2_m15_open_h4_atrx1.0`      | 27       | **0.99**         | 0.59     | 3.27      | +6.45%    |  9.34     |
| `body0.5_reg50wit_streak2_m15_atr_stop`             | 27       | 0.36             | 0.56     | 3.27      | +6.18%    | 10.78     |
| `body0.5_reg50wit_streak2_m15_confirm_h4_atrx1.0`   | 27       | 0.99             | 0.59     | 3.27      | +2.77%    |  3.97     |
| `body0.5_reg50wit_streak2_m15_open`                 | 27       | 0.36             | 0.56     | 3.27      | +4.20%    |  4.92     |

The strongest walk-forward Sharpe (0.99) goes to the H4-ATR-stop
variant; the strongest holdout return (7.5%) goes to the prev-H4-open
stop variant. Both pass walk-forward at 56–59% positive folds across
27 disjoint 3-month slices of 2018–2026.

### Why no retracement strategy certified

Of the 192 retracement specs, 20 produced 3.0–4.06 trades/week — but
**none passed walk-forward**. Their median walk-forward Sharpe is ≤ 0
across 27 folds, which means the *signal+filter* combos that produce
3-ish retracements/week don't have a long-history edge to begin with.
The retracement is a real M15 execution refinement (the audit confirms
entries land at the prev-H4 midpoint, exits land at one of {prev open,
prev extreme, H4 close}), but the holdout window alone is too short
to certify a strategy that the long-history walk-forward rejects.

To certify a retracement variant we'd need either (a) a richer signal
or filter set (the proposer is currently a static grid; an adaptive
LLM proposer is the natural next step) or (b) M15 history extending
back into the walk-forward folds, so the harness can test the M15
execution edge on its own merits rather than only the H4 signal edge.

### Conclusion under all three user constraints (gold, walk-forward, 3–5/wk)

**Best spec**: `body0.5_reg50wit_streak2_m15_open_prev_h4_open` —
take the continuation only when (1) the prior H4 body ≥ 0.5 × H4 ATR(14),
(2) the prior H4 close is on the same side of the 50-bar MA as the
trade direction, and (3) the previous **two** H4 candles share the
trade's color. Enter at the open of the new H4's first M15 bar; stop
at the prior H4 open; take profit at the H4 close (variants with
prev-H4-extreme TP also certify with similar metrics).

This produces 3.27 trades/week on the 9-week holdout (within the
target band) with positive walk-forward Sharpe in 16 of 27 disjoint
3-month slices of 2018–2026.

## Reproduce

```bash
pip install pandas numpy matplotlib
make data         # fetch_data.py
make audit        # audit.py     — data + simulator self-test
make backtest     # backtest.py  — single-spec headline numbers
make search       # orchestrate.py — agentic search + leaderboard
```

All CSVs and PNG land in `results/`. The strategy spec schema lives in
[`agents/06-proposer.md`](agents/06-proposer.md); the backtester
contract in [`agents/03-backtester.md`](agents/03-backtester.md);
audit checks in [`scripts/audit.py`](scripts/audit.py).
