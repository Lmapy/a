# Agent 28 — Prop Certifier

## Role

Replaces the alpha-pipeline certifier (`validation/certify.py`) for
prop-firm purposes. A combination is **prop-certified** if AND ONLY IF:

```
pass_probability         >= 0.35
first_payout_probability >= 0.20
blowup_probability       <= 0.15
consistency_breach_rate  <= 0.15
median_days_to_pass      <= 30
works on multiple account models  (verified by leaderboard)
```

Implemented in `prop_challenge.score.passes_cert`. The runner writes
every passing combo to `results/certified_prop_challenge_strategies.json`.

## Why these thresholds

| metric | rationale |
| --- | --- |
| `pass >= 35%` | Topstep/MFFU charge ~$50–$200 per challenge attempt; below 35% the expected cost of repeated attempts exceeds the first payout. |
| `payout >= 20%` | even after passing, ~80% of funded accounts blow before first payout in industry data. 20% is a 4× improvement on baseline. |
| `blowup <= 15%` | challenge fees + funded-account fees compound; a 15% ceiling keeps expected drawdown survivable across attempts. |
| `consistency <= 15%` | every prop firm publishes a 30–50% single-day rule. 15% gives margin against MC noise. |
| `median_days <= 30` | longer challenges mean more chance of macro shock; 30 days covers ~6 NFPs / ~1 FOMC. |

## Multi-account requirement

A combo that passes only on one account is over-fitted to that firm's
specific drawdown geometry. The certifier requires the same combination
(strategy × risk_model × daily_rules) to also certify on at least one
other account; the multi-account check is implicit in the leaderboard
filtering logic.

## Output

`results/certified_prop_challenge_strategies.json` — full schema with
all sim numbers, breach histograms, and recommended account choice.
