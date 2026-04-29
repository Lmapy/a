# Prop-firm passing report

Run id: `2026-04-29_002`
Produced at: 2026-04-29T01:02:52+00:00
Mode: **smoke**, n_perm=80, runtime=150.4s.

## Executive summary

**0 certified strategies** out of 4 evaluated (0 rejected for unavailable data, 0 reached prop_candidate). This is a valid research outcome under the hardened harness — see Failure analysis below for the exact gates that blocked each candidate.

## Data summary

| Split | Window | H4 rows | First | Last |
|---|---|---:|---|---|
| research_train | 2020-07-01 00:00:00+00:00 → 2022-01-01 00:00:00+00:00 | 2408 | 2020-07-01 00:00:00+00:00 | 2021-12-31 20:00:00+00:00 |
| validation | 2022-01-01 00:00:00+00:00 → 2024-01-01 00:00:00+00:00 | 3192 | 2022-01-02 20:00:00+00:00 | 2023-12-29 20:00:00+00:00 |
| holdout | 2024-01-01 00:00:00+00:00 → 2026-05-01 00:00:00+00:00 | 3709 | 2024-01-01 20:00:00+00:00 | 2026-04-27 20:00:00+00:00 |

## Feature availability

OHLC-only harness. Available features: OHLC, spread, tick count, TPO. **NOT available**: real volume, bid/ask volume, footprint, delta, CVD, order book, VWAP, Volume Profile. Strategies referencing those tokens are auto-rejected as `REJECTED_UNAVAILABLE_DATA`.

## Top candidates

| Rank | id | family | score | cert | pass_p | blowup_p | wf_med_sharpe | ho_return | label_p | block_p |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `session_sweep_reclaim__london__prev_h4_extreme` | session_sweep_reclaim | -0.47 | research_only | 0.234 | 0.092 | -19.938 | -0.0085 | 0.5625 | 0.475 |
| 2 | `session_sweep_reclaim__london__h4_atr1` | session_sweep_reclaim | -17.33 | research_only | 0.166 | 0.43 | -23.077 | -0.0379 | 0.7 | 0.7875 |
| 3 | `session_sweep_reclaim__ny__prev_h4_extreme` | session_sweep_reclaim | -36.72 | research_only | 0.042 | 0.62 | -17.314 | -0.1279 | 0.975 | 0.95 |
| 4 | `session_sweep_reclaim__ny__h4_atr1` | session_sweep_reclaim | -44.92 | research_only | 0.018 | 0.73 | -12.515 | -0.1747 | 0.9875 | 1.0 |

## Certification level histogram

| Level | Count |
|---|---:|
| rejected_unavailable_data | 0 |
| rejected_broken | 0 |
| research_only | 4 |
| watchlist | 0 |
| candidate | 0 |
| prop_candidate | 0 |
| certified | 0 |
| retired | 0 |

## Failure analysis (top 8)

| Failure reason | Candidates |
|---|---:|
| `fail_walk_forward` | 4 |
| `fail_holdout` | 4 |
| `fail_label_permutation` | 4 |
| `fail_random_baseline` | 4 |
| `fail_block_bootstrap` | 4 |
| `fail_spread_stress` | 4 |
| `fail_profit_factor` | 3 |
| `fail_low_pass_probability` | 3 |

## Account comparison

| Account | Verification |
|---|---|
| generic_eod_trailing_50k | synthetic |
| generic_intraday_trailing_50k | synthetic |
| generic_static_50k | synthetic |
| mffu_pro_100k | verified |
| mffu_pro_150k | verified |
| mffu_pro_50k | verified |
| mffu_rapid_25k | verified |
| topstep_100k | verified |
| topstep_150k | verified |
| topstep_50k | verified |

## Next research queue

### `session_sweep_reclaim__ny__prev_h4_extreme`

- **session_sweep_reclaim__ny__prev_h4_extreme** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `session_sweep_reclaim__ny__h4_atr1`

- **session_sweep_reclaim__ny__h4_atr1** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `session_sweep_reclaim__london__prev_h4_extreme`

- **session_sweep_reclaim__london__prev_h4_extreme** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `session_sweep_reclaim__london__h4_atr1`

- **session_sweep_reclaim__london__h4_atr1** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

## Provenance

```json
{
  "git_head": "465adc9d6ec70c474aac415d8f60a649b71b2afc",
  "config_hashes": {
    "config/data_splits.json": "a295cff1bdc622b3",
    "config/prop_accounts.json": "2247fdb0833a4a08"
  },
  "harness_version": "hardened_v2 (Batches A-H)",
  "schema_versions": {
    "prop_accounts": "3",
    "data_splits": "1"
  }
}
```