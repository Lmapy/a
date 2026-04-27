# Agent 15 — alpha spec builder (agent 03 in v3 spec)

## Role

Convert each hypothesis into one or more concrete, executable specs by
expanding the family template against its variants. Each spec is then
checked for runnability:

- the spec's signal/filter/exit/stop types must be implemented in the
  executor
- the entry model must be compatible with the requested entry timeframe
  (`entry_models.compatibility.ENTRY_MODEL_TIMEFRAME_MAP`)
- the entry timeframe must have on-disk bar data
  (`AVAILABLE_ENTRY_TIMEFRAMES = {"M15"}` today)

A spec that fails any of those gets `status = "skipped"` with explicit
`skip_reasons` instead of silently dropping. **No spec is ever
fabricated to pass.**

## Output

```json
{
  "n_total": 40,
  "n_runnable": 16,
  "n_skipped": 24,
  "available_entry_timeframes": ["M15"],
  "specs": [
    { "spec": { ... full spec ... },
      "family_id": "fib_continuation",
      "entry_timeframe": "M15",
      "compatibility_status": "ok",
      "status": "runnable",
      "skip_reasons": [] }
  ]
}
```

Written to `results/generated_specs.json`.

## Why so many skips?

Today we only have **M15** data on disk. Several families prefer M5
sub-bars (`sweep_reclaim`, `minor_structure_break`); those specs are
correctly emitted as `skipped: data_unavailable` rather than degraded
to M15.

## Run

```bash
python3 scripts/agent_03_spec_builder.py
```
