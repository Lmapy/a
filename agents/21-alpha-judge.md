# Agent 21 — Alpha Judge (meta-agent)

## Role

Read every result file the rest of the pipeline produced and emit one
structured verdict at `results/alpha_judge.json`. Deterministic — no
LLM call, no narrative generation. The narrative comes from rules over
the leaderboard, critic report, certified strategies file, and prop
simulation.

## Inputs

- `results/leaderboard.csv`
- `results/critic_report.json`
- `results/certified_alpha_strategies.json`
- `results/prop_simulation.json`

## Output schema

```json
{
  "verdict": "no_alpha_certified" | "alpha_candidate_with_low_confidence" | "multiple_alpha_candidates" | "no_results",
  "verdict_detail": "...",
  "patterns": {
    "n_evaluated": 16,
    "n_passes_critic": 3,
    "n_certified": 0,
    "edge_dies_under_stress": 4,
    "edge_survives_stress": 12,
    "shuffle_p_above_0.05": 16,
    "random_p_above_0.05": 16,
    "wf_below_60pct_positive_folds": 12,
    "below_3_trades_per_week": 8,
    "above_5_trades_per_week": 1,
    "pf_above_1.2": 4,
    "pf_below_1.0": 5
  },
  "critic_failure_histogram": {"longest losing streak ...": 3, ...},
  "common_failures": ["edge dies under spread", ...],
  "recommendations": [
    "longer M15 history is the single biggest unlock",
    "add session filter (NY-only)",
    ...
  ],
  "near_misses": [ ... up to 3 closest candidates with reasons ... ]
}
```

## Rules behind `recommendations`

| condition | recommendation |
| --- | --- |
| 0 certified | "no strategy survived all gates; see near_misses" |
| shuffle p>0.05 on EVERY spec | "edge has no time-structure; sample size is the bottleneck" |
| random-baseline p>0.05 on ≥90% of specs | "selectivity needs to rise OR sample size" |
| ≥70% of specs below 60% positive folds | "edge is regime-dependent; try regime_class filter" |
| ≥30% of specs trade <3/wk | "loosen filters to bring trade count up" |
| ≥30% of specs trade >5/wk | "tighten filters to reduce noise" |
| any spec flips negative under stress | "edge not real-world tradable; reduce frequency or use limits" |
| top critic failure has count ≥3 | flag it directly |

Always appended:

> "single biggest unlock: more *same-broker* M5 history. Sub-M15
> execution-model improvements cannot be fairly tested until that
> lands. See `results/data_manifest.json:known_gaps`."

## Run

Embedded in `scripts/run_alpha.py`. Stand-alone:

```bash
python3 scripts/agent_alpha_judge.py
```

## Why deterministic, not LLM

The judge is a checklist over structured numbers. Letting an LLM
generate the verdict would re-introduce subjective bias the rest of
the pipeline carefully removes. If the user wants narrative, that
belongs in the README or the audit PDF — never in the audit chain.
