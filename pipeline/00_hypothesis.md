# 00 — Hypothesis Definition

## Core idea
At each 4-hour candle open, trade in the same direction as the previous 4-hour candle body:
- Previous candle close > open → long next 4-hour candle.
- Previous candle close < open → short next 4-hour candle.
- Do nothing for doji candles (close == open).

## Why this might work
- Short-horizon momentum / order-flow persistence.
- Session handoff continuation (Asia→Europe→US).
- Trend periods where directional autocorrelation is positive.

## Failure modes
- Mean-reverting/choppy regimes.
- News-driven reversals at session boundaries.
- Transaction costs erasing thin edge.
