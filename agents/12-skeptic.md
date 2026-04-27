# Agent 12 — skeptic

## Role

Find what the search missed. For each top certified champion in the
leaderboard, runs three deterministic probes and writes the results to
`results/skeptic.csv`. The output is structured (one row per probe, no
free-form prose) so the leaderboard can be filtered against it later.

## Probes

### 1. Perturbation — fragility under parameter nudges

Runs the champion at single-knob deltas:

- `cost_bps` ∈ {0.0, 0.5, 1.0, 2.0, 3.0, 5.0}
- `body_atr.min` ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.2}
- `body_atr.atr_n` ∈ {7, 10, 14, 21}
- `regime.ma_n` ∈ {20, 30, 50, 80, 100, 150}
- `entry.level` (fib only) ∈ {0.236, 0.382, 0.5, 0.618, 0.786}
- `stop` ∈ {none, prev_h4_open, prev_h4_extreme, h4_atr×1.0}

For each variant we re-run walk-forward + holdout and report whether
the variant **still certifies** under the same rule the orchestrator
uses. The biggest answer this delivers: how much spread cost the
strategy can absorb before the edge dies (the *cost break-even*).

### 2. Attribution — which filter is doing the work

Drops one filter at a time and re-runs walk-forward. The Sharpe drop
is each filter's marginal contribution. Also drops *all* filters as a
sanity floor — that should always reproduce the raw-signal Sharpe.

### 3. Coverage — what the grid skipped

Generates near-miss specs that the orchestrator's grid did not
enumerate:

- `regime.ma_n` at 30, 80, 100 (grid only had 50)
- `body_atr.min` at 0.3 and 0.7 (grid only had 0.5 and 1.0)
- Add a streak filter (k=2, k=3) where the champion lacks one
- Add the regime filter where the champion lacks one
- Combine fib retracement entries with a streak filter (the original
  retracement grid did not cross these)

These get the same walk-forward + holdout treatment so a coverage hit
that *would* have certified shows up directly as a "still_certified =
True" row outside the original grid.

## Inputs

- `results/leaderboard.csv` (must have `certified == True` rows).
- The same data files as the rest of the pipeline.

## Outputs

- `results/skeptic.csv` — one row per probe with columns:
  `champion_id, probe_type, probe_label, spec_id, spec_json,
   wf_folds, wf_median_sharpe, wf_pct_positive_folds, wf_avg_total_return,
   ho_trades, ho_trades_per_week, ho_total_return, ho_sharpe_ann,
   still_certified`.

## Run

```bash
make skeptic
```

## How to read the output

For each champion the script prints:

- **cost break-even** — the highest `cost_bps` at which certification
  still holds, and the first value at which it breaks. Anything below
  ~2 bps is a fragile edge — real spread is usually higher than that.
- **per-filter Δ** — the wf-Sharpe change when that filter is removed.
  Big negative Δ = load-bearing filter. Δ near zero = decoration.
- **coverage hits** — any off-grid variant whose `still_certified` is
  True. These are concrete suggestions the search should have found
  but didn't.

## Acceptance checks (run by the agent itself)

- The baseline row for each champion must reproduce the leaderboard's
  walk-forward and holdout numbers exactly (within rounding).
- Dropping all filters must produce wf-Sharpe ≤ baseline.
- Cost perturbation results must be monotonic in cost (higher cost ⇒
  weakly worse wf-Sharpe). Violations are real bugs, not noise.
