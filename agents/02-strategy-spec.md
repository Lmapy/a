# Agent 02 — strategy-spec

## Hypothesis

> On every 4-hour candle open, gold (XAUUSD) tends to continue in the same
> direction as the previous 4-hour candle.

This document is the **single source of truth** for the trading rules.
Whenever the rules change here, `scripts/backtest.py` must be updated to
match.

## Signal (4-hour timeframe)

For an H4 bar `t` with open `O_t` and close `C_t`, define its **color**:

```
color_t = sign(C_t - O_t)   ∈ {-1, 0, +1}
```

The trade direction at `t` is the previous bar's color:

```
sig_t = color_{t-1}
```

If `sig_t == 0` (previous bar was a doji) the trade is skipped.

## Entry (15-minute timeframe)

Three execution variants are tested. Direction is `sig_t`; size is unit
notional (`+1` for buy, `-1` for sell).

1. **`m15_open`** — Enter at the open of the first M15 sub-bar inside H4 bar
   `t`. Equivalent to a market order at the H4 open price; serves as the
   sanity baseline.

2. **`m15_confirm`** — Walk forward through the M15 sub-bars of `t` and
   enter at the **close** of the first sub-bar whose own color equals
   `sig_t`. If no confirming bar appears within `t`, no trade is taken.

3. **`m15_atr_stop`** — Same entry as `m15_open`, but additionally place a
   protective stop one M15-ATR(14) (computed on the M15 series) opposite the
   trade direction. The stop is checked bar-by-bar against M15 high/low.

## Exit

- For variants 1 and 2: exit at the close of the H4 bar `t`.
- For variant 3: exit at whichever comes first — the stop being touched on
  any M15 bar, or the close of H4 bar `t`.

## Costs

Round-trip spread is approximated as `(spread_entry + spread_exit) * point`,
where `point = 0.001` (XAUUSDc point size from the broker spec). Subtracted
in price units from the trade's gross PnL.

## What the strategy does NOT do

- No leverage, no compounding across trades — each trade is reported as a
  return on its entry price.
- No position-sizing logic; the pipeline is intentionally a clean test of
  whether the *direction* signal has any edge.
- No stop based on H4 ATR or H4 highs/lows; the stop in variant 3 is
  deliberately the M15-ATR to keep the spec consistent with "M15 entries".

## What we want to know

For each variant: trade count, win rate, mean trade return (bps),
total return, annualised Sharpe, max drawdown, and average hold time. These
are computed in `scripts/backtest.py` and written to `results/summary.csv`.

In addition, before any execution: **what is the empirical probability that
an H4 bar continues the previous H4 bar's direction, on the long history?**
That is computed in Stage 1 and written to `results/hit_rate.csv`.
