# Agent 17 — robustness critic (agent 05 in v3 spec)

## Role

Try to **kill** every promising strategy. Run a fixed set of checks on
the trade ledger; if any fires, the strategy is rejected at the critic
layer regardless of its v2 certification status.

## Tests (in `validation/critic.py:run_critic`)

1. **top_trade_dependency**
   - remove the single top trade by abs return; total return must stay
     positive
   - remove top 5% by abs return; total return must stay positive
   - single biggest winner must be ≤ 40% of all winning PnL
2. **worst_week** — worst rolling 7-day window must be ≥ −10%
3. **consecutive_loss** — longest losing streak ≤ 6 trades
4. **session_split** — strategy must be profitable in ≥ 2 of
   {asia, london, ny}
5. **removed_top_year** — with the single best calendar year removed
   from the long history, remainder PnL must still be positive

## Output (per spec)

```json
{
  "strategy_id": "...",
  "passes_critic": false,
  "failure_reasons": [
    "profit disappears after removing top 1 trade (0.0541 -> -0.0021)",
    "longest losing streak 8 > 6 max"
  ],
  "detail": { ... raw numbers per check ... }
}
```

Aggregated across all specs in `results/critic_report.json`.

## Why this exists

The v2 certifier already requires `biggest_trade_share ≤ 0.20`, but
that's only one cut. Real prop-firm rules care about **drawdown
clusters**, not just single-trade dependence — a strategy can survive
the v2 gates and still detonate in one bad week. The critic catches
those cases.
