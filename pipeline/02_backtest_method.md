# 02 — Backtest Methodology

## Signal
`signal_t = sign(close_{t-1} - open_{t-1})`

## Trade construction
- Enter at `open_t`.
- Exit at `close_t`.
- Return: `signal_t * (close_t/open_t - 1)`.

## Metrics
- Win rate
- Mean return per trade
- Total compounded return
- Approx annualized Sharpe
- Max drawdown
- Benchmark (always long each 4-hour bar)

## Required robustness next
- Add fees/slippage.
- Regime split by volatility.
- Walk-forward or rolling out-of-sample tests.
