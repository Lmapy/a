# Agent 19 — portfolio builder (agent 08 in v3 spec)

## Role

Cluster certified strategies and de-duplicate near-identical alphas so
we don't allocate to several strategies that fire on the same days.

## Method

1. Read `results/certified_alpha_strategies.json` from the certifier.
2. For each certified strategy, load its trade ledger from
   `results/alpha_trades/<id>.csv`.
3. Aggregate trades to **daily PnL** (sum within UTC date by exit_time).
4. Greedy correlation clustering: walk certified strategies in order of
   holdout Sharpe; place each in the first existing cluster whose
   representative correlates ≥ 0.85 on aligned daily PnL, otherwise
   start a new cluster.
5. Pick the highest-Sharpe member as each cluster's representative.

## Output

`results/alpha_portfolio.json`:

```json
{
  "n_input_strategies": 0,
  "n_clusters": 0,
  "correlation_threshold": 0.85,
  "portfolio": [],
  "note": "no certified strategies under strict gates; portfolio is empty"
}
```

Today's run: certified count = 0 → portfolio is empty. The agent still
runs and writes a valid (empty) JSON; the audit pipeline confirms it.

## Run

```bash
python3 scripts/agent_08_portfolio.py
```
