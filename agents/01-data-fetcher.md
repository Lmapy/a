# Agent 01 — data-fetcher

## Role

Pull real XAUUSD OHLC bars from the public sources listed below and write
normalised CSVs into `data/`. Never fabricate, simulate, or interpolate
prices. If a source is unreachable, fail loudly — do not invent data.

## Inputs

None (the URLs are pinned in `scripts/fetch_data.py:SOURCES`).

## Outputs

| File                          | Bars     | Span                          | Use                                 |
| ----------------------------- | -------- | ----------------------------- | ----------------------------------- |
| `data/XAUUSD_H4_long.csv`     | ~8,600   | 2018-06-28 → 2026-04-20       | Stage-1 long-history hit-rate study |
| `data/XAUUSD_H4_matched.csv`  | ~260     | 2026-01-30 → 2026-04-01       | Stage-2 H4 signal generation        |
| `data/XAUUSD_M15_matched.csv` | ~3,977   | 2026-01-30 → 2026-04-01       | Stage-2 M15 entry execution         |

All files share the schema `time, open, high, low, close, volume, spread`,
with `time` parsed as UTC and rows sorted ascending with duplicates removed.

## Sources

- **Long H4 history.** `github.com/142f/inv-cry`, file
  `data_2015_xau_btc/processed/mt5/H4/XAUUSDc.csv`. MT5 broker export.
- **Matched M15 + H4.** `github.com/tiumbj/Bot_Data_Basese`, files
  `Local_LLM/dataset/tf/XAUUSD_H4.csv` and `…/XAUUSD_M15.csv`. Same broker
  for both timeframes — required so M15 sub-bars line up cleanly inside H4
  buckets.

## Implementation

`scripts/fetch_data.py`. Run it directly:

```bash
python3 scripts/fetch_data.py
```

## Acceptance checks

- All three output files exist, are non-empty, and parse with
  `pandas.read_csv(..., parse_dates=["time"])`.
- The H4 and M15 matched files cover the same date span (within a couple of
  bars at the edges).
- For each H4 bar in the matched file, ≥ 1 M15 bar exists in the same 4h
  bucket. (`backtest.py:simulate` reports `diag_skipped_no_subbars` and
  this should be 0 in a healthy run.)
