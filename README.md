# Gold 4H Candle Continuation Research Pipeline

This repo contains a reproducible workflow to test the idea:

> On each new 4-hour candle open, price is likely to continue in the same direction as the previous 4-hour candle.

It now includes a walk-forward optimizer that explicitly targets **3–5 trades per week**.

## Quickstart

```bash
python3 scripts/download_gold_data.py --symbol GC=F --interval 60m --range 730d --out data/gold_1h.csv
python3 scripts/backtest_4h_continuation.py \
  --input data/gold_1h.csv \
  --output-trades results/gold_4h_trades.csv \
  --output-report results/backtest_report.md
python3 scripts/walkforward_optimize.py \
  --input data/gold_1h.csv \
  --train-days 180 \
  --test-days 30 \
  --min-trades-per-week 3 \
  --max-trades-per-week 5 \
  --output-folds results/walkforward_folds.csv \
  --output-report results/walkforward_report.md
```

Then read:
- `pipeline/*.md` for process and research plan
- `results/backtest_report.md` for baseline metrics
- `results/walkforward_report.md` for walk-forward OOS results
- `agents/*.md` for role-based agent prompts you can reuse
