# Agent: Walk-Forward Optimizer

Goal: discover robust parameter sets that hold up out-of-sample and stay near 3–5 trades/week.

Checklist:
1. Define train/test rolling windows.
2. Run parameter search on train folds only.
3. Penalize trade frequency outside the weekly target.
4. Apply selected params to test fold (no re-tuning on test).
5. Export fold-level CSV and markdown summary.
