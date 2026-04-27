# Agent 10 — auditor

## Role

Self-test the data and the simulators. The auditor is the safety net
that catches silent bugs before they pollute the leaderboard. Every
finding is either a **PASS** (printed for visibility) or a **FAIL**
(also counted; the script returns non-zero on any failure so it can be
wired into CI).

## Inputs

- `data/XAUUSD_H4_long.csv`
- `data/XAUUSD_H4_matched.csv`
- `data/XAUUSD_M15_matched.csv`

## Checks performed

### 1. Data integrity

For each dataset:

- non-empty
- timestamps are monotonic, unique, and tz-aware UTC
- OHLC has no NaNs and no non-positive prices
- `high == max(open, high, close, low)` and `low == min(...)` row-wise
- `≥ 80 %` of consecutive deltas land on the expected step minutes
  (240 for H4, 15 for M15) — with a tolerant > on weekend and holiday
  gaps
- no zero or backwards time deltas

### 2. Cross-dataset alignment (H4 matched ↔ M15 matched)

- every H4 bar has ≥ 1 M15 sub-bar in its 4-hour bucket
- no orphan M15 buckets outside the H4 matched range
- median M15 bars per bucket is 16 (the ideal tile)
- minimum bucket size ≥ 4 (so any spec that needs a confirmation bar
  has somewhere to look)

### 3. Simulator correctness

- `compute_signal_np` puts a 0 at index 0 and `color[i-1]` at index `i`
  for `i ≥ 1`. **Look-ahead probe**: perturbing `close[i]` does not
  change `sig[i]`.
- a synthetic *leaked* oracle signal beats the real signal by a wide
  margin (proves the harness can tell signals apart, and our real
  signal isn't accidentally leaking).
- pick a real H4 bar, replay it manually, and assert that
  `run_full_sim` produces the same direction, entry, exit, and return
  to within `1e-9`. **This is the check that caught the cost-model
  bug**: when the manual replay used `cost_model: bps`, `cost_bps: 0`
  but the runner was secretly using broker spread, the returns
  disagreed.
- retracement strategy: a `m15_retrace_50` spec must produce trades
  whose entry equals the previous H4 midpoint, whose direction matches
  the previous H4 color, and whose exit is one of {prev H4 open
  (stop), prev H4 high/low (TP), current H4 close (time exit)}.
- diagnostics counters returned by `run_full_sim(..., return_diag=True)`
  are non-negative integers and sum to a count consistent with the
  H4 input length.

### 4. Walk-forward harness

- ≥ 10 folds
- non-overlapping test windows
- each fold has `train_end == test_start`
- a known certified spec produces trades in ≥ 80 % of folds (smoke
  test: empty folds suggest a slicing or warmup bug)

## Run

```bash
python3 scripts/audit.py        # exit 0 on success, 1 on any failure
make audit
```

The full report is also written to `results/audit.txt`.

## Why this exists

Trading research is uniquely vulnerable to silent bugs. A negative bug
("I'm dropping 10 % of bars without telling you") looks the same as a
weak signal: the strategy just doesn't make money, and you'll spend
weeks tuning a strategy that was being starved of data the whole time.
A positive bug (look-ahead leak) looks the same as a brilliant
strategy: the equity curve goes up, and you'll deploy something that
will lose money in production.

The auditor is cheap to run and catches both classes before the
leaderboard ever sees them.
