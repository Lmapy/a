# Agent 24 — Risk Sizing Optimiser

## Role

Find the per-trade sizing model that maximises challenge pass
probability without breaching drawdown buffers. Implemented in
`prop_challenge/risk.py`.

## Models tested (per account)

| name | rule |
| --- | --- |
| `micro_1` | always 1 contract |
| `micro_2` | always 2 contracts |
| `dollar_risk_50` | size to risk ~$50 per trade |
| `dollar_risk_100` | size to risk ~$100 per trade |
| `pct_dd_buffer_2pct` | size to risk 2% of remaining drawdown buffer |
| `reduce_after_loss` | base 2 contracts, halve after a losing trade |
| `scale_after_high` | base 1 contract, +1 after each new equity high |

Every model is capped at `min(account.max_contracts, 5)`.

## Hard guarantee

`risk.size()` always returns ≥ 1 (no zero sizing) and ≤ contract cap.
`pct_dd_buffer` saturates at the cap, so drawdown buffer cannot be
silently exceeded.

## Output

Aggregated in `results/prop_risk_model_comparison.csv`:

| risk_model | n | median_pass | median_blowup | median_payout | median_score |

Highest `median_score` wins; ties broken by lower `median_blowup`.
