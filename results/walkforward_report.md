# Walk-Forward Optimization Report (Gold 4H Direction Model)

## Objective
Optimize parameters with an explicit trade-frequency target of **3.0 to 5.0 trades/week**.

## Aggregate Out-of-Sample
- **folds**: 1
- **avg_oos_trades_per_fold**: 2.000000
- **avg_oos_trades_per_week**: 2.800000
- **avg_oos_total_return_per_fold**: 0.003423
- **oos_trade_frequency_target_met_ratio**: 0.000000

## Fold Highlights
- **Fold 1** 2026-01-15T00:00:00+00:00 → 2026-01-22T00:00:00+00:00
  - params: mode=continuation, hours=[0, 8, 16], min_prev_body_pct=0.0015, fee=0.0
  - weekdays (Mon=0): [0, 2, 4]
  - OOS: trades/week=2.800, total_return=0.003423, sharpe=383.714357