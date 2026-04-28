# 01 — Data Specification

## Instrument
- Gold continuous futures proxy via Yahoo symbol `GC=F`.

## Required columns
- `timestamp` (UTC)
- `open`, `high`, `low`, `close`
- `volume` (optional for this baseline)

## Frequency
- Source: 1-hour bars.
- Backtest timeframe: resampled 4-hour bars.

## Validation checks
- Monotonic timestamps.
- No duplicate timestamps.
- No missing OHLC within used periods.
