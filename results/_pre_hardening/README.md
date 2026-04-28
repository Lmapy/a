# Pre-hardening results

Files in this directory predate the harness hardening pass (Batches
A–D, see `docs/HARDENING_REPORT.md`). They are kept for archeology
**only** and should not be cited as evidence for any strategy.

The four most consequential reasons these results are not trustworthy:

1. **Spread costs were undercharged ~1000×.** The Dukascopy codec
   stores `spread = ask − bid` in price units, but the executor
   multiplied that value by `POINT_SIZE = 0.001` before charging it.
   Every Sharpe / total-return / pass-probability number on this
   branch pre-fix is therefore inflated.

2. **Walk-forward and holdout tested different strategies.** Walk-
   forward silently rewrote any M15 entry model to `h4_open` while
   holdout used the real M15-aware executor. The "certified" label
   only applied to the H4-open variant, not the M15 variant a live
   trader would deploy.

3. **Prop-firm risk sizing leaked future PnL into position size.**
   `prop_challenge.risk.RiskModel.size()` accepted `trade_pnl_price`,
   the realised outcome of the trade about to be sized, and computed
   per-contract risk as `abs(trade_pnl_price * dpp)`. Pass-probability
   estimates from this code path are unreliable upward.

4. **The "shuffle test" gate was a no-op.** Sharpe is invariant to
   per-trade permutation, so `shuffled_outcome_test` always returned
   `p ~= 1.0`. Specs that "passed" the shuffle gate did so by
   accident.

Regenerate trustworthy leaderboards with the canonical hardened
pipeline:

```bash
make pipeline    # writes results/leaderboard_hardened.csv + .meta.json
```

The provenance sidecar (`*.meta.json`) carries the git HEAD, config
hashes, split coverage, and prop-account verification status.
