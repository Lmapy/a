# Prop Firm Challenge Trading Algorithm

Multi-strategy algorithmic trading system designed to pass prop firm challenges (Topstep, My Funded Futures) on 50k futures accounts with EOD trailing drawdown.

## Target Firms & Rules

| Parameter | Value |
|-----------|-------|
| Account Size | $50,000 |
| Profit Target | $3,000 |
| Trailing Drawdown | $2,000 (EOD) |
| Instruments | ES, NQ, MES, MNQ |

**EOD trailing drawdown** means the high-water mark only updates at end of day. Intraday unrealized losses don't count against the drawdown — only the closing balance matters.

## Architecture

```
Risk Engine (gates every order)
    ├── EOD Trailing Drawdown Tracker
    ├── Dynamic Position Sizer (scales with drawdown budget)
    └── Lockout Manager (daily loss limit, tilt prevention, gain protection)

Meta Layer (selects strategy based on market conditions)
    ├── Regime Detector (ADX + Bollinger Bands + EMA)
    └── Strategy Selector (regime → strategy mapping)

Strategies
    ├── Mean Reversion (VWAP fade in ranging markets)
    ├── Trend Following (EMA pullback in trending markets)
    └── Breakout (opening range break in volatile markets)

Backtesting
    ├── Event-driven backtest engine
    ├── Challenge simulator
    └── Monte Carlo pass rate estimator (block bootstrap)
```

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scipy pyyaml matplotlib yfinance pytest

# Download real futures data
python scripts/download_data.py

# Run a backtest on real ES data
python scripts/run_backtest.py --instrument ES --timeframe 5min

# Run with CSV data
python scripts/run_backtest.py --instrument ES --data-dir ./data

# Simulate multiple challenge attempts
python scripts/run_challenge_sim.py --n-runs 10

# Monte Carlo pass rate estimation
python scripts/run_monte_carlo.py --n-trials 10000
```

## Configuration

All parameters are in YAML config files under `config/`:

- `config/firms/topstep_50k.yaml` — Firm rules (drawdown, target, max contracts)
- `config/strategies/*.yaml` — Strategy parameters (entry/exit thresholds, ATR multiples)
- `config/global.yaml` — Risk management settings (daily loss limit, position sizing)

## How It Works

### Risk Management (Most Critical)

The risk engine is the core of the system. Every order passes through it:

1. **EOD Drawdown Tracking**: High-water mark updates only at session close. We exploit this by tolerating wider intraday stops.
2. **Dynamic Position Sizing**: Trade bigger when ahead (acceleration mode), smaller when behind (protection mode), stop completely when near the floor.
3. **Lockout Rules**: Stop trading after 3 consecutive losses, daily loss exceeding budget threshold, or to protect afternoon gains.

### Strategy Selection

The meta layer detects the current market regime and activates the appropriate strategy:

| Regime | Primary Strategy | Allocation |
|--------|-----------------|------------|
| Strong Trend | Trend Following | 70% |
| Weak Trend | Trend Following + Mean Reversion | 50/50 |
| Ranging | Mean Reversion | 80% |
| High Volatility | Breakout | 60% |
| Low Volatility | Mean Reversion | 100% |

### Monte Carlo Simulation

Uses block bootstrap resampling of daily P&Ls to estimate:
- **Pass rate** with 95% confidence interval
- **Average days to pass**
- **Risk of ruin**
- **Optimal daily profit target**

## Project Structure

```
├── config/          # YAML configuration files
├── src/
│   ├── risk/        # Risk engine, drawdown tracker, position sizer
│   ├── strategy/    # Mean reversion, trend following, breakout
│   ├── meta/        # Regime detection, strategy selection
│   ├── execution/   # Order management, paper broker
│   ├── backtest/    # Backtest engine, challenge sim, Monte Carlo
│   ├── data/        # Data models, feeds, indicators
│   └── core/        # Orchestrator, config, trade journal
├── scripts/         # Run scripts
└── tests/           # Unit tests
```

## Testing

```bash
python -m pytest tests/ -v
```
