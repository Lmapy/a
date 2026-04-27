# Agent: Backtest Analyst

Goal: run baseline continuation strategy and publish metrics.

Checklist:
1. Resample to 4H OHLC.
2. Build previous-candle-direction signal.
3. Compute trade-level returns and equity curve.
4. Output:
   - `results/gold_4h_trades.csv`
   - `results/backtest_report.md`
