# Agent 13 — alpha data auditor (agent 01 in the v3 spec)

## Role

Verify the data the rest of the alpha pipeline reads. The agent does
*not* run any strategy — it only confirms the bars are real, complete,
aligned, and pinned to a known commit SHA before any further agent
runs.

## Inputs

- `data/XAUUSD_H4_long.csv`
- `data/XAUUSD_H4_matched.csv`
- `data/XAUUSD_M15_matched.csv`
- `results/data_manifest.json` (written by `scripts/fetch_data.py`)

## Outputs

- `results/agent_data_audit.json` with:
  - `validator_findings`: every rule-level finding from
    `data.validator.run_full_validation` (severity, code, message)
  - `manifest_present`: whether the manifest exists
  - `manifest`: full pinned-source list

## Rules enforced (every one is a hard fail)

- timestamps tz-aware UTC
- monotonic, deduped
- aligned to bar boundary
- valid OHLC (low≤open,close,high; high≥open,close,low)
- M15 sub-bars must each map to a known H4 bucket
- resampled M15→H4 within 5×point of broker H4
- every source URL must contain a commit SHA, not `/HEAD/`

If any of these fail, the runner aborts before agent 02 fires.

## Run

```bash
python3 scripts/run_alpha.py     # agent 01 runs first inside this script
```

Or directly:

```bash
python3 -c "from scripts.run_alpha import agent_01_data_audit; \
  from data.loader import load_all; agent_01_data_audit(load_all())"
```
