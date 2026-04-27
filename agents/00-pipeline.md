# 4H Continuation on Gold — agent pipeline

This folder describes the pipeline as a sequence of named agents. Each agent
has a single responsibility, a clear input contract, a clear output contract,
and points at the concrete script that implements it. The agents can be run
sequentially via `make all` (see top-level `Makefile`) or individually.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐
│ 01 data-fetcher │ →  │ 02 strategy-spec │ →  │ 03 backtester   │ →  │ 04 analyst     │ →  │ 05 reporter     │
│ pull real OHLC  │    │ define rules     │    │ simulate trades │    │ compute stats  │    │ write up result │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └────────────────┘    └─────────────────┘
        │                       │                       │                      │                       │
   data/*.csv             agents/02-*.md          results/trades.csv    results/summary.csv      README.md
```

## Hard rules

1. **No synthetic data.** Every price bar used in this pipeline must come from
   a real broker or exchange. Both data sources used here are public MT5
   exports on GitHub:
   - long-history H4: `github.com/142f/inv-cry`
   - matched H4 + M15: `github.com/tiumbj/Bot_Data_Basese`
2. **No look-ahead.** A trade decision at time `t` may only use bars closed
   strictly before `t`.
3. **Costs are real.** Round-trip spread is taken from the M15 broker feed
   (in points, multiplied by point size) and subtracted from each trade's PnL.
4. **Backtest before tuning.** The hit-rate diagnostic must be computed and
   reported before any strategy parameter is tuned.

## Run the pipeline

```bash
python3 scripts/fetch_data.py          # 01
# (02 is documentation only; rules live in scripts/backtest.py)
python3 scripts/backtest.py            # 03 + 04
# 05 is README.md, regenerated from results/*.csv when needed
```
