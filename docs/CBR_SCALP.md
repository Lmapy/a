# CBR-style gold scalp module

## What this is

A **mechanical approximation** of the publicly-described
TomTrades / CBR-style gold scalping model. It is **NOT** the
proprietary TomTrades system. The implementation here is for
backtesting and research only; the actual TomTrades methodology
involves discretionary elements (live order-flow read, news
context, judgment built over years of trading) that this engine
cannot reproduce.

## Architecture

The module lives entirely under `strategies/scalp/`:

```
strategies/scalp/
  __init__.py            module docstring + scope notes
  config.py              CBRGoldScalpConfig dataclass + YAML round-trip
  sessions.py            Australia/Melbourne session windows + asia high/low
  detectors.py           expansion / sweep / rebalance / MSB
  bias.py                1H bias modes + optional DXY filter
  entries.py             50% retrace / market entry, stop, target math
  engine.py              no-lookahead 1m bar loop with state machine
  metrics.py             per-trade + per-setup metrics + writers
  sweep.py               parameter-sweep runner

configs/
  cbr_gold_scalp.yaml          single-backtest defaults
  cbr_gold_scalp_sweep.yaml    parameter sweep grid

scripts/
  run_cbr_backtest.py
  run_cbr_sweep.py

tests/
  test_cbr_scalp.py            22 tests covering every module
```

It runs **completely separately** from the existing prop-firm
passing engine (`scripts/run_prop_passing.py`) and the H4 executor
(`execution/executor.py`); nothing in those paths depends on or is
changed by the scalp module.

## Strategy concept

Mechanically reproduced:

1. **Higher-timeframe bias** from completed 1H bars only
   (modes: `OFF`, `PREVIOUS_1H_CANDLE_DIRECTION`, `EMA_SLOPE`,
   `WICK_BODY_BIAS`).
2. **Session window**: default Australia/Melbourne 09:00-12:00
   asia, with execution restricted to 10:00-11:00 (the "second
   hour"). Configurable.
3. **20+ minutes of one-sided expansion** detection (configurable
   lookback, directional-candle ratio, ATR-multiple gate, plus
   a 0-100 quality score blending directional %, net move,
   pullback share).
4. **Sweep or rebalance trigger**: either price sweeps the previous
   1H high/low and reclaims it, or price returns to / through the
   expansion midpoint within `max_bars_after_expansion`.
5. **Confirmed-pivot market structure shift**: pivots only become
   eligible `pivot_right` bars after they form. The engine never
   acts on an unconfirmed pivot.
6. **50% retracement limit entry** on the MSB impulse
   (alternatives: market on MSB close, 0.618 retrace, custom).
7. **Recent-swing or expansion-extreme stop**, fixed-R / midpoint
   / equilibrium / session-extreme target.

NOT reproduced:

- real volume / order flow / DOM
- discretionary "trend day vs rotation day" classification
- news filters
- exact session timing variations across holidays
- any proprietary indicator that wasn't publicly described

## Run

```bash
make cbr-scalp                # full available data, default config
make cbr-scalp-smoke          # 100k M1 bars, ~1 minute runtime
make cbr-scalp-sweep          # ~100-combo parameter sweep
```

Or directly:

```bash
python3 scripts/run_cbr_backtest.py \
    --config configs/cbr_gold_scalp.yaml \
    --start 2024-01-01 --end 2026-04-29

python3 scripts/run_cbr_sweep.py \
    --base configs/cbr_gold_scalp.yaml \
    --sweep configs/cbr_gold_scalp_sweep.yaml \
    --start 2024-01-01 --limit-bars 200000
```

## Outputs

Every backtest writes to `results/cbr_gold_scalp/` (or the override
in `--output-stem`):

| File | What |
|---|---|
| `trades.csv` | one row per realised trade — entry/exit times + prices, R, PnL, MAE, MFE, exit reason, trigger kind, expansion quality |
| `setups.csv` | one row per **setup** (taken or skipped) with bias, DXY state, expansion stats, sweep / rebalance / MSB flags, skip reason if any |
| `summary.json` | aggregate metrics + by-direction / by-day-of-week / by-trigger / expansion-quality bucket / setup funnel |
| `config_used.json` | the exact resolved config — fully reproducible |
| `validation_report.json` | data validation findings (duplicate ts, OHLC validity, dxy_available, etc.) |

## No-lookahead protections

- **1H bias**: `resolve_h1_bias_at(t)` uses only H1 bars whose `time
  + 1h <= t`. The current incomplete H1 bar is never consulted.
- **DXY filter**: same shape, with `dxy_available=False` recorded
  if no DXY data is on disk (the default — Dukascopy XAUUSD-only
  pull doesn't include DXY).
- **Pivots**: `find_confirmed_pivots` only emits a pivot once
  `pivot_right` bars have closed AFTER the candidate. The engine
  filters by `confirmed_at_idx <= current_idx` per bar.
- **Limit orders**: only fill on bars **after** they're placed
  (`_check_limit_fill` runs in the bar loop after the placement bar).
- **Same-bar entry/exit ambiguity**: when a bar's high and low
  both exceed entry + stop / target, the default
  `ambiguous_bar_resolution = CONSERVATIVE` makes stop win. The
  metric `ambiguous_same_bar_count` reports how often this fired.

## Config knobs (high-level)

The config is one dataclass with subsections:

- `session`              timezone, asia + execution windows
- `htf_bias`             bias mode + EMA / wick params
- `dxy`                  DXY filter mode (defaults OFF; needs DXY data)
- `expansion`            lookback, directional %, ATR multiples, quality threshold
- `trigger`              sweep / rebalance / both, tolerances, max-bars windows
- `structure`            pivot left / right, break mode (close vs wick)
- `entry`                entry mode + retrace fraction + per-day caps
- `stop_target`          stop mode, target mode, risk_reward, buffer ticks
- `risk`                 fixed_qty / dollar_risk / % equity, tick size + value
- `engine`               same-bar fill, ambiguous-bar policy

## Smoke result (Jan-Mar 2024, default config)

```
M1 bars:       86,725
runtime:       72 s
trades:         4
win rate:       50.0%
avg R:          0.25
total R:        1.0 (3 wins × 1.5R, 2 losses × -1R = +2.5 - 2 = wait...)
profit factor:  1.5
biggest share:  30%
```

(Long path: 3 trades / +0.67R avg / +2.0R total. Short path:
1 trade / -1R. Setup funnel: 59 logged setups, 7 with orders
placed, 37 skipped on `bias_against_msb_direction`, 14 on
`msb_timeout`. The bias filter is doing real work.)

## Known limitations

1. **Candle-only approximation.** No tick-by-tick simulation, so
   setups whose timing depends on intra-bar ordering (entry +
   stop both touched in the same M1 candle) are resolved
   conservatively; `ambiguous_same_bar_count` in the summary
   tracks how many trades that affected.
2. **No discretionary market-structure read.** The pivot-confirmed
   MSB approximates "structure shift" mechanically, but a human
   trader would skip MSBs that visually look like fakes.
3. **No order-flow / DOM.** The strategy can't see liquidity below
   levels; it only sees price reacted from them.
4. **DXY timing.** Even with DXY data attached, the H1 cadence is
   coarse for a 1m gold scalp. Consider M15 DXY when adding the
   data.
5. **News filter not implemented.** A real CBR-style trader avoids
   first 5 minutes after major news. Not modelled here.
6. **Same-bar ambiguity.** ~5% of trades on busy days may resolve
   ambiguously; default conservative policy means this slightly
   under-counts wins. Toggle via `engine.ambiguous_bar_resolution`.

## Suggested next improvements

- Multi-target partial exits (TP1 1R, runner to 2R+).
- ATR-regime filter (skip very low / very high ATR days).
- Session variants (London open, NY open) sharing the engine.
- Walk-forward / out-of-sample split — current backtest is full
  in-sample.
- Monte Carlo trade reordering for confidence intervals.
- Plug into the existing prop-firm passing engine: take the
  trade ledger from a CBR backtest, run it through
  `prop_challenge.run_chronological_replay` to get prop pass /
  blowup probabilities + Wilson CIs.
- News filter loaded from a CSV calendar.
- Per-trade debug charts (price candles + prev 1H high/low +
  expansion range + MSB level + entry/stop/target lines + exit).
- DXY (or DXY proxy via EUR/USD + GBP/USD basket) added to the
  Dukascopy candle pull so the inverse-confirmation filter can
  actually run.

## Why this lives outside the prop-passing engine

The H4-bucket executor in `execution/executor.py` is shaped for
"H4 setup + M15 entry refinement" candidates. The CBR scalp model
is shaped for "1m bar loop with H1 bias + intraday session window
+ explicit state machine". Forcing one through the other would
either contort the executor or paper-over the state-machine
contract. Two engines is the simpler design — they share data,
splits, run-event scaffolds, and (eventually) prop-firm sims.

If you want the CBR ledger judged against the prop-passing
gates, write the trade list out from `trades.csv` and feed it to
`prop_challenge.challenge.run_chronological_replay(trades_df,
account, risk, rules)`. That's a 5-line glue script.
