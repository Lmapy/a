# Prop-firm passing report

Run id: `2026-04-29_004`
Produced at: 2026-04-29T01:38:25+00:00
Mode: **default**, n_perm=500, runtime=1136.7s.

## Executive summary

**0 certified strategies** out of 165 evaluated (0 rejected for unavailable data, 0 reached prop_candidate). This is a valid research outcome under the hardened harness — see Failure analysis below for the exact gates that blocked each candidate.

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
| 1 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__risk_fixed_micro_1` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 2 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__risk_fixed_micro_2` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 3 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__risk_dollar_risk_50` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 4 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__risk_dollar_risk_100` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 5 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__risk_pct_dd_buffer_2pct` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 6 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__risk_reduce_after_loss` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 7 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__risk_scale_after_high` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 8 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__rules_none` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 9 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__rules_max1` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |
| 10 | `previous_h4_range_retracement__fib0.618__prev_h4_extreme__rules_max2` | previous_h4_range_retracement | 12.496 | research_only | None | None | 6.684 | 0.0477 | 0.216 | 0.12 |

## Certification level histogram

| Level | Count |
|---|---:|
| rejected_unavailable_data | 0 |
| rejected_broken | 6 |
| research_only | 159 |
| watchlist | 0 |
| candidate | 0 |
| prop_candidate | 0 |
| certified | 0 |
| retired | 0 |

## Failure analysis (top 8)

| Failure reason | Candidates |
|---|---:|
| `fail_label_permutation` | 165 |
| `fail_random_baseline` | 165 |
| `fail_block_bootstrap` | 165 |
| `fail_holdout` | 89 |
| `fail_spread_stress` | 89 |
| `fail_walk_forward` | 62 |
| `fail_profit_factor` | 34 |
| `fail_low_pass_probability` | 7 |

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

### `opening_range_breakout_retest__ny_open__prev_h4_open`

- **opening_range_breakout_retest__ny_open__prev_h4_open** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `opening_range_breakout_retest__ny_open__h4_atr1`

- **opening_range_breakout_retest__ny_open__h4_atr1** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `opening_range_breakout_retest__london_open__prev_h4_open`

- **opening_range_breakout_retest__london_open__prev_h4_open** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `opening_range_breakout_retest__london_open__h4_atr1`

- **opening_range_breakout_retest__london_open__h4_atr1** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `opening_range_failed_breakout__ny_open__prev_h4_extreme`

- **opening_range_failed_breakout__ny_open__prev_h4_extreme** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

### `opening_range_failed_breakout__ny_open__h4_atr1`

- **opening_range_failed_breakout__ny_open__h4_atr1** — REJECT_AND_REDESIGN: failed random / label permutation gate; the directional edge is indistinguishable from random. Suggest moving to a different strategy family.

## Provenance

```json
{
  "git_head": "bfa486d6e558ae42dbe081e28a1445179b73ab52",
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