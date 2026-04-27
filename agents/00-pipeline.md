# 4H Continuation on Gold — agent pipeline

This folder describes the pipeline as a sequence of named agents. Each agent
has a single responsibility, a clear input contract, a clear output contract,
and points at the concrete script that implements it.

There are two flows. The **single-spec flow** (agents 01–05) tests one
fixed strategy and produces the README's headline numbers. The **search
flow** (agents 06–09) runs an agentic loop that proposes many specs,
gates each through walk-forward + holdout, and writes a leaderboard.

## Single-spec flow (`make all`)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐
│ 01 data-fetcher │ →  │ 02 strategy-spec │ →  │ 03 backtester   │ →  │ 04 analyst     │ →  │ 05 reporter     │
│ pull real OHLC  │    │ define rules     │    │ simulate trades │    │ compute stats  │    │ write up result │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └────────────────┘    └─────────────────┘
        │                       │                       │                      │                       │
   data/*.csv             agents/02-*.md          results/trades.csv    results/summary.csv      README.md
```

## Search flow (`make search`)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌────────────────┐
│ 06 proposer     │ →  │ 07 walk-forward  │ →  │ 08 critic       │ →  │ 09 orchestrator│
│ emit JSON spec  │    │ ~25 rolling folds│    │ certify or not  │    │ leaderboard.csv│
└─────────────────┘    └──────────────────┘    └─────────────────┘    └────────────────┘
        ↑                                                                       │
        └──────────── reads leaderboard for next round ─────────────────────────┘
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
# single fixed-spec flow
python3 scripts/fetch_data.py          # 01
python3 scripts/backtest.py            # 03 + 04 (rules quoted in 02)

# agentic search flow
python3 scripts/orchestrate.py         # 06 + 07 + 08 + 09
```

Outputs of the search flow:

- `results/leaderboard.csv` — ranked candidates with walk-forward and
  holdout metrics, and a `certified` flag.
- `results/search_folds.csv` — per-fold detail across all candidates.
- `results/search_holdout_trades.csv` — every holdout trade across all
  candidates so any leaderboard row can be audited end-to-end.
