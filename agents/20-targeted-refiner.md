# Agent 20 — Targeted Refiner (replaces v3 placeholder agent 06)

## Role

Pick the top-N **near-miss** strategies from the leaderboard and
generate single-knob structured refinements. Each refinement is
re-evaluated through the same gates the original runner uses.

## Near-miss definition

A strategy that **passes the robustness critic** but fails one or both
of the statistical gates (shuffled-outcome p>0.05, random-baseline
p>0.05) — or fails the 60% positive-folds gate by ≤5 pp. These are the
candidates with real-looking trade structure that just need either
sample-size or a better filter to clear strict cert.

## Allowed knobs (one per round)

| label | patch |
| --- | --- |
| `session_NY` | add `session` filter (UTC hours 12, 16) |
| `session_London` | add `session` filter (UTC hours 8, 12) |
| `pdh_pdl_inside` | add `pdh_pdl` filter (inside mode) |
| `htf_vwap_dist_2.5` | add `htf_vwap_dist` filter (≤2.5σ) |
| `regime_trend` | add `regime_class` filter (allow=trend) |
| `regime_trend_expansion` | add `regime_class` filter (allow=trend, expansion) |
| `atr_pct_30_80` | add `atr_percentile` band 30%–80% |
| `stop_h4_atr_x1` | swap stop to `h4_atr` ×1.0 |
| `stop_prev_h4_open` | swap stop to `prev_h4_open` |
| `stop_prev_h4_extreme` | swap stop to `prev_h4_extreme` |
| `body_min_0.3` | tighten/relax `body_atr.min` |
| `body_min_0.7` | tighten/relax `body_atr.min` |
| `streak_2`, `streak_3` | add `min_streak` filter |
| `fib_0.382`, `fib_0.618`, `fib_0.705` | swap `fib_limit_entry.level` |

## Forbidden

- adding more than one knob in a round
- adding a knob the parent already has (skipped, not stacked)
- referencing the holdout p-values when picking the next knob
- using future bars

## Outputs

- `results/refinement_log.json`
- `results/refined_specs.csv` — one row per (parent, knob) variant
  with full evaluation metrics and `certified` boolean

## Run

Embedded in `scripts/run_alpha.py`. Stand-alone:

```bash
python3 scripts/agent_06_refiner.py
```
