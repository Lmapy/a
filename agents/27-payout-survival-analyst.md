# Agent 27 — Payout Survival Analyst

## Role

A passing strategy is worthless if it never reaches first payout or
self-detonates immediately after going funded. This agent runs the
funded-account simulator (`prop_challenge.payout.simulate_payout`) on
every strategy × account that passed the challenge sim and reports:

- `first_payout_probability` — share of funded paths that hit
  `payout_target` over `payout_min_days` without breach
- `blowup_before_payout_probability` — share of funded paths that
  blow up before reaching first payout
- `consistency_breach_rate` (on funded) — strict 50% rule violations
- `median_days_to_payout`

## How it differs from the challenge sim

Same drawdown rules, but the exit condition is `payout_target` instead
of `profit_target`, and the day budget is the firm's `max_days_to_payout`
window rather than `max_challenge_days`.

## Output

`results/prop_payout_survival.csv` (best of any rule/risk per
strategy×account), plus a `first_payout_probability` column on the
main leaderboard so you can sort by it directly.

## Why this matters

Many "high pass-prob" strategies are actually death-spiral martingales
in disguise: they hit profit_target via two big winning days but
detonate on average funded paths because they trade too aggressively.
Filtering by payout survival kills those false positives.
