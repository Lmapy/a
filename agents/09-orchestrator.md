# Agent 09 — orchestrator

## Role

The conductor. Pulls a batch of specs from the proposer, runs each one
through walk-forward and holdout, asks the critic to certify, and writes
everything to the leaderboard.

## Loop

```
1. specs = proposer.propose()                       # agent 06
2. for spec in specs:
       wf = walkforward(spec, h4_long)              # agent 07
       ho = full_sim(spec, h4_matched, m15_matched) # agent 03 (full sim)
       certified = critic.certify(wf, ho)           # agent 08
       leaderboard.append({...})
3. write leaderboard.csv, search_folds.csv, search_holdout_trades.csv
```

## Outputs

- `results/leaderboard.csv` — one row per spec with its walk-forward
  aggregates, holdout metrics, and `certified` flag.
- `results/search_folds.csv` — per-fold detail across all specs.
- `results/search_holdout_trades.csv` — every holdout trade across all
  specs (so you can audit any row of the leaderboard).

## Run

```bash
python3 scripts/orchestrate.py
```

This is intentionally a single command — the whole agentic search
collapses into one entry point so you can wire it under `make search`
or behind a CI button later.

## Future extensions

- **Adaptive proposer.** Replace `propose()` with a Claude API call that
  reads the leaderboard and emits novel specs. The proposer should be
  given walk-forward columns only, not holdout columns, to prevent
  selection bias on the holdout.
- **Parallel evaluation.** Specs are independent; an executor pool lets
  you evaluate dozens at once.
- **Champion model.** After N rounds, persist the top certified spec to
  `results/champion.json` and use it as the basis for an actual paper-
  trading run.
