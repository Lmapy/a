# Run events / progress scaffold

The hardened pipeline writes per-run progress and events into
`results/runs/<run_id>/`:

```
results/runs/2026-04-29_001/
  progress.json     # aggregate live state, atomically rewritten
  events.jsonl      # append-only one-event-per-line audit trail
  summary.json      # written once at the end of a run
```

A future visual research control room (Batch J) will tail
`events.jsonl` and read `progress.json` to render per-stage progress.

## When to use it

The scaffold is **optional**. The hardened pipeline (Batches A–E)
runs unchanged whether or not an `EventWriter` is constructed; the
helper functions `emit_candidate(None, ...)` and
`emit_stage(None, ...)` are no-ops when no writer is supplied.

Use it from a runner that wants to surface progress to the future
UI:

```python
from pathlib import Path
from core.run_state import RunState
from core.run_events import EventWriter, emit_candidate, emit_stage

state = RunState.create(Path("results/runs"))
events = EventWriter(state)

state.set_stage("strategy_generator")
emit_stage(events, stage="strategy_generator", status="running")

for cand in candidates:
    state.bump("generated")
    emit_candidate(events,
                    candidate_id=cand.id, family=cand.family,
                    stage="ohlc_backtest", status="running")
    # ... evaluate ...
    if rejected_unavailable_data:
        state.bump("rejected_unavailable_data")
        emit_candidate(events,
                        candidate_id=cand.id, family=cand.family,
                        stage="feature_capability_auditor",
                        status="rejected",
                        certification_level="rejected_unavailable_data",
                        failure_reasons=["rejected_unavailable_data"],
                        message="VWAP-required strategy")
        continue
    # ...

state.set_status("certified")
state.write_summary({"git_head": "...", "leaderboard": "results/x.csv"})
```

## Allowed vocabulary

`core/run_state.py:ALLOWED_STAGES`:

```
strategy_generator
feature_capability_auditor
ohlc_backtest
entry_model_lab
walk_forward
validation
holdout
risk_sweep
daily_rule_optimiser
prop_firm_simulator
robustness_critic
judge
leaderboard
report
```

`core/run_state.py:ALLOWED_STATUSES`:

```
queued running passed failed warning rejected
watchlist candidate prop_candidate certified skipped
```

Any value outside these sets raises a `ValueError` immediately, so
the events stream stays machine-parseable.

## Event schema

Each event is one JSON object on its own line. Required keys are
`stage` and/or `status`; everything else is optional. The writer
auto-stamps `timestamp` and `run_id`.

```json
{
  "timestamp": "2026-04-29T10:42:01+00:00",
  "run_id": "2026-04-29_001",
  "candidate_id": "london_sweep_reclaim_017",
  "family": "session_sweep_reclaim",
  "stage": "walk_forward",
  "status": "failed",
  "certification_level": "research_only",
  "failure_reasons": ["fail_walk_forward"],
  "metrics": {"wf_pct_positive": 0.42, "wf_median_sharpe": -0.18},
  "message": "Candidate failed walk-forward gate."
}
```

## progress.json schema

`progress.json` is rewritten atomically after every state mutation.
A UI can `stat()` it on a poll interval and only re-render when the
file's mtime changes.

```json
{
  "run_id": "2026-04-29_001",
  "started_at": "2026-04-29T10:40:00+00:00",
  "updated_at": "2026-04-29T10:42:01+00:00",
  "status": "running",
  "current_stage": "walk_forward",
  "counts": {
    "generated": 120,
    "rejected_unavailable_data": 8,
    "rejected_broken": 0,
    "backtested": 90,
    "walk_forward_passed": 14,
    "walk_forward_failed": 76,
    "holdout_passed": 0,
    "holdout_failed": 0,
    "candidates": 0,
    "prop_candidates": 2,
    "certified": 0,
    "failed": 74,
    "skipped": 0
  },
  "active_candidates": ["london_sweep_reclaim_017"],
  "recent_events": [{"timestamp": "...", "candidate_id": "...",
                     "stage": "walk_forward", "status": "failed",
                     "certification_level": "research_only"}]
}
```

## When to call `write_summary`

Once per run, at the end. Pass any extras the future UI / report
generator will want (commit SHA, paths to leaderboards, runtime).
The summary lives at `results/runs/<run_id>/summary.json`.

## Forward compatibility

Adding new optional fields to events is safe — JSON consumers
ignore unknown keys. Adding new allowed stages or statuses requires
updating `ALLOWED_STAGES` / `ALLOWED_STATUSES` in `core/run_state.py`
(and optionally the corresponding test); the writer will reject
unknown values until the vocabulary is widened explicitly.

The full UI design (network view, leaderboard, candidate detail,
failure analysis, account comparison, risk/daily-rule optimiser,
TPO/session explorer) is **Batch J** and is not yet implemented.
This scaffold is the contract Batch J will read.
