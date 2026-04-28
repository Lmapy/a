# Crypto Futures Research Pipeline (Python 3.11+)

## Concise implementation plan (as requested)
1. **Files created**
   - Modular package under `src/` for adapters, features, strategies, backtest, and utilities.
   - Config at `config/settings.yaml`, tests in `tests/`, dependency lock-in via `requirements.txt`.
2. **Data endpoints used**
   - **Bybit V5**: `/v5/market/kline`, `/v5/market/funding/history`, `/v5/market/open-interest`, `/v5/market/mark-price-kline`, `/v5/market/index-price-kline`.
   - **Hyperliquid** info endpoint `POST /info` with `candleSnapshot` and `fundingHistory` payloads.
3. **Canonical schema**
   - `timestamp, exchange, symbol, timeframe, open, high, low, close, volume, funding_rate, open_interest, mark_price, index_price`.
4. **Assumptions**
   - UTC timestamps everywhere.
   - Missing public OI/mark/index on Hyperliquid are stored as nulls with warnings.
   - All costs (fees/slippage/leverage/funding impact/risk controls) are config-driven.

## Setup
```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI workflow
```bash
python -m src.main download --exchange bybit --symbols BTCUSDT ETHUSDT SOLUSDT --start 2023-01-01 --end 2026-01-01
python -m src.main download --exchange hyperliquid --symbols BTC ETH SOL --start 2023-01-01 --end 2026-01-01
python -m src.main features
python -m src.main backtest --strategy breakout
python -m src.main backtest --strategy sweep
python -m src.main backtest --strategy vwap_reversion
python -m src.main portfolio
```

## Architecture
- `src/adapters`: Exchange-specific API adapters behind common interface.
- `src/features`: Canonical feature engineering and session-level tags.
- `src/strategies`: Strategy signal generation with event tags.
- `src/backtest`: Event-driven backtest engine, metrics, and chart generation.
- `src/utils`: Config/logging/time/io helpers.

## Reproducibility and data handling
- Raw and processed datasets are saved as Parquet under `data/raw` and `data/processed`.
- HTTP caching is enabled for Bybit GET requests (request hash cache).
- Chunked downloads and timestamp deduplication are built in.
- Candle continuity checks raise warnings.

## Notes for extending Binance later
- Add `src/adapters/binance.py` implementing the `ExchangeAdapter` methods.
- Wire adapter registration in `src/adapters/factory.py`.
- No strategy changes required.
