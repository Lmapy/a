# Hyperliquid Multi-Asset Trading Bot

Automated trading bot for Hyperliquid perpetuals that scans all available assets, applies a hybrid strategy combining momentum, mean-reversion, and liquidity signals, and manages risk across the entire portfolio.

## Features

- **All-asset scanning**: Reads every perp market on Hyperliquid, filters by volume
- **Hybrid strategy**: Combines momentum (EMA/MACD/ROC), mean-reversion (RSI/Bollinger), liquidity (orderbook imbalance), and funding rate signals
- **CoinGlass integration**: Optional liquidity data including long/short ratios, OI changes, and liquidation levels
- **Risk management**: Per-position sizing, portfolio-level drawdown limits, spread checks, orderbook depth limits
- **Automatic SL/TP**: ATR-based stop loss and take profit with trigger orders
- **Dry-run mode**: Analyze markets and see signals without executing trades

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your private key and settings
```

## Usage

```bash
# Dry run - see signals without trading
python main.py --dry-run

# Single cycle
python main.py --once

# Live trading
python main.py

# Debug logging
python main.py --log-level DEBUG
```

## Configuration

All settings are in `.env`. Key parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `HL_PRIVATE_KEY` | required | Your wallet private key |
| `HL_WALLET_ADDRESS` | required | Your wallet address |
| `HL_MAINNET` | false | true for mainnet, false for testnet |
| `COINGLASS_API_KEY` | optional | CoinGlass API key for liquidity data |
| `MAX_LEVERAGE` | 3 | Maximum leverage per position |
| `MAX_POSITIONS` | 15 | Maximum concurrent positions |
| `MAX_DRAWDOWN_PCT` | 10 | Emergency close-all drawdown threshold |
| `STRATEGY_MODE` | hybrid | momentum, mean_reversion, liquidity, or hybrid |
| `REBALANCE_INTERVAL_SECONDS` | 300 | Seconds between rebalance cycles |

## Architecture

```
hyperliquid_bot/
  config.py        - Environment-based configuration
  market_data.py   - Hyperliquid API + CoinGlass data fetching
  strategy.py      - Multi-signal strategy engine
  risk_manager.py  - Position sizing and risk limits
  portfolio.py     - Portfolio management and execution
  bot.py           - Main orchestrator
main.py            - CLI entry point
```

## Risk Warning

This bot trades real money. Always test on testnet first. Use conservative settings. Never risk more than you can afford to lose. Past performance does not guarantee future results.
