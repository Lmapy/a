# 04 — Walk-Forward Workflow (Target 3–5 Trades/Week)

## Purpose
Use rolling train/test windows to avoid overfitting and to explicitly target a practical trade cadence.

## Step-by-step
1. Build 4H bars from hourly gold data.
2. Split into repeated folds:
   - Train window (e.g., 180 days)
   - Test window (e.g., 30 days)
3. On each train fold, grid-search parameters:
   - Direction mode: continuation vs reversal.
   - Entry-hour subset (controls trade frequency).
   - Minimum previous-body size filter.
   - Per-trade cost assumption.
4. Select the best train parameters using score:
   - Reward Sharpe + compounded return.
   - Penalize trade frequency outside **3–5 trades/week**.
5. Freeze parameters and evaluate only on the test fold.
6. Aggregate out-of-sample results across folds.

## Why this matches your requirement
- Entry-hour selection and body-size filters are direct levers for weekly trade count.
- The optimizer penalizes models that do not stay close to 3–5 trades/week.

## Run command
```bash
python3 scripts/walkforward_optimize.py \
  --input data/gold_1h.csv \
  --train-days 180 \
  --test-days 30 \
  --min-trades-per-week 3 \
  --max-trades-per-week 5 \
  --output-folds results/walkforward_folds.csv \
  --output-report results/walkforward_report.md
```
