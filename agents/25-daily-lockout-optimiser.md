# Agent 25 — Daily Lockout Optimiser (CRITICAL)

## Role

The single biggest lever in prop trading is **knowing when to stop for
the day**. This agent enumerates daily-rule sets and finds the
combination that minimises drawdown and rule breaches without crushing
pass probability. Implemented in `prop_challenge/lockout.py`.

## Rule sets tested

| name | rule |
| --- | --- |
| `none` | no lockout |
| `max1` / `max2` / `max3` | cap on trades per day |
| `stop_w1` | lock after 1 winning trade |
| `stop_l1` / `stop_l2` | lock after 1 / 2 losing trades |
| `dp250` / `dp500_dl300` | dollar-based daily profit / loss stops |
| `dl250` | daily loss stop only |
| `ny_only_max2` | only trade NY session, cap 2 |
| `london_only_max2` | only trade London session, cap 2 |

## How "session_only" interacts with the executor

`lockout.session_in()` reuses `regime.filters.session_label()` so the
session definitions are consistent with the rest of the pipeline (no
two definitions of "NY"). `london_open` and `ny_open` add the first 2
hours of those sessions for explicit "open-only" tests.

## Output

Aggregated in `results/prop_daily_rules_comparison.csv`. Best combo
per (strategy, account) lands in
`results/prop_account_comparison.csv`. The full search lives in
`results/prop_challenge_leaderboard.csv` so any combination can be
audited end-to-end.
