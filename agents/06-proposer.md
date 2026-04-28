# Agent 06 — proposer

## Role

Generate the next strategy spec to evaluate. The proposer reads the
existing leaderboard, decides what's worth trying next, and emits a
single JSON spec that the backtester can run without further translation.

## Output: strategy spec (JSON)

```json
{
  "id": "body0.5_ses12-16_h4_atrx2.0_m15_open",
  "signal": {"type": "prev_color"},
  "filters": [
    {"type": "body_atr",  "min": 0.5, "atr_n": 14},
    {"type": "session",   "hours_utc": [12, 16]},
    {"type": "regime",    "ma_n": 50, "side": "with"},
    {"type": "min_streak","k": 2}
  ],
  "entry": {"type": "m15_open"},
  "stop":  {"type": "h4_atr", "mult": 2.0, "atr_n": 14},
  "exit":  {"type": "h4_close"},
  "cost_bps": 1.5
}
```

### Filter types currently supported

| `type`        | meaning                                                                        | params                   |
| ------------- | ------------------------------------------------------------------------------ | ------------------------ |
| `body_atr`    | only trade when the previous H4 body / its ATR ≥ `min`                         | `min`, `atr_n`           |
| `session`     | only trade when the H4 bar opens at one of the listed UTC hours                | `hours_utc` (list[int])  |
| `regime`      | require previous-bar close to be above (`with`) or below (`against`) MA(`ma_n`)| `ma_n`, `side`           |
| `min_streak`  | require the previous `k` H4 bars to share the signal direction                 | `k`                      |

### Entry / stop choices

- `entry.type` ∈ `m15_open`, `m15_confirm`, `m15_atr_stop`, `h4_open`
- `stop.type` ∈ `none`, `h4_atr`, `m15_atr` (with `mult`, `atr_n`)
- `exit.type` ∈ `h4_close` (only one for now)

## Implementation today

`scripts/orchestrate.py:propose()` is a deterministic Cartesian grid
over the filter / entry / stop choices. It produces ~324 unique specs.
This is an honest baseline — but it is a grid, not an agent.

## Implementation later (the actual agent)

Replace `propose()` with a function that:

1. Reads `results/leaderboard.csv` (sorted by `wf_median_sharpe` desc).
2. Calls Claude with a structured prompt: "here are the top-K strategies
   tried so far and their metrics — propose N new specs that are
   *different* from what's been tried (in JSON), aiming to improve
   walk-forward Sharpe without overfitting the holdout."
3. Validates each returned spec against the schema before adding it.
4. Bans specs whose normalised id already exists in the leaderboard.

The spec schema is the contract between the proposer and the rest of the
pipeline. Anything the proposer can express, the backtester can run.

## Acceptance checks

- Every emitted spec round-trips through `json.dumps`/`json.loads`
  unchanged.
- `spec_id(spec)` is unique within a single search run.
- The proposer never emits a spec that was certified in a previous run
  (we don't waste compute re-validating known winners).
