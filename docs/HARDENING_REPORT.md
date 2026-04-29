# Hardening report — Batches A through H

Date: 2026-04-29
Branch: `claude/4h-candle-strategy-backtest-8kKdx`
Pre-fix commit: `f9437b6360f0e31639d93fbc1c4ee3a48addeffc`

> **Status:** harness hardening landed end-to-end. **Audit: 0
> failures.** Tests: 75 / 75 passing across 11 files. Old leaderboards
> are archived under `results/_pre_hardening/` and explicitly marked
> invalid; the canonical pipeline (`make pipeline`) writes
> `results/leaderboard_hardened.csv` with a `.meta.json` provenance
> sidecar listing the git HEAD, config hashes, and prop-account
> verification statuses.
>
> Real-firm prop accounts (Topstep 50/100/150K, MyFundedFutures
> Rapid 25K + Pro 50/100/150K) re-verified 2026-04-29 against current
> public rule pages. MFFU restructured 2025/2026; old keys
> `mffu_25k/50k/100k/150k` renamed to `mffu_rapid_25k`,
> `mffu_pro_50k/100k/150k`. Schema extended to allow `null`
> `daily_loss_limit` (MFFU 2026 has no DLL) and `null`
> `max_challenge_days` (untimed evaluations).

## Batch A summary

| # | Phase | Status | What changed |
|---|-------|--------|--------------|
| 1 | Spread unit fix | ✅ landed | `execution/executor.py`, `scripts/strategy.py` no longer multiply `spread` by `POINT_SIZE`. Spread is stored as `ask − bid` in price units at the codec; cost per leg is the value directly. Old code under-charged by ~1000×. |
| 2 | Split loader + tests | ✅ landed | New `data/splits.py` exposing `load_splits()` returning train/validation/holdout for both H4 and M15. Half-open windows, ordering check, overlap rejection, holdout-leak guard, coverage diagnostic. |
| 3 | No-lookahead | ✅ landed | `_apply_filters` now shifts `atr_percentile`, `pdh_pdl`, `vwap_dist`, `htf_vwap_dist` by one bar. Mutation regression tests verify entries are insensitive to future bars but responsive to prior bars. |
| 4 | Test gate | ✅ landed | `make all` and `make report` run `make test` first. New `scripts/run_tests.py` for Windows hosts without GNU make. Test files: `test_executor`, `test_registry`, `test_compatibility`, `test_stop_exit_spread`, `test_validator`, `test_splits`, `test_no_lookahead`. |
| 5 | Audit expansion | ✅ landed | `scripts/audit.py` now fails on: split coverage gap, spread×POINT_SIZE in canonical paths, walk-forward M15 downgrade, prop-sim `trade_pnl_price` leak, shuffle-test still wired, leaderboards without provenance metadata. |

## Detailed changes

### Phase 4 — spread unit fix

**Bug.** Dukascopy candles store `spread = ask − bid` in **price units**:
the codec at [data/_dukascopy_codec.py](../data/_dukascopy_codec.py) lines
255–256 scales `ask` and `bid` by `1/price_scale` (= POINT_SIZE) before
computing the diff. Per-bar `spread_mean` is therefore $/oz on XAUUSD.

The executor at [execution/executor.py](../execution/executor.py) lines
381 and 469 was further multiplying that spread by `POINT_SIZE = 0.001`,
producing a $0.0003 per-leg cost where the broker charges ~$0.30. Same
pattern in [scripts/strategy.py](../scripts/strategy.py) line 624.

**Fix.** Removed the `* POINT_SIZE` rescale. Spread per leg is now
`sub["spread"].iloc[i]` directly. `POINT_SIZE` is retained as the unit
constant for *near-miss tolerance* arithmetic (1 tick = $0.001), which
is its only legitimate use.

**Test.** [tests/test_stop_exit_spread.py](../tests/test_stop_exit_spread.py)
rewritten with realistic price-unit fixtures (entry $0.30, exit-bar
spike $0.80, tail $0.20). Adds:

- `test_spread_unit_from_bid_ask` — assert cost equals (ask − bid) at
  entry + (ask − bid) at exit, lock the codec convention.
- `test_spread_mult_scales_cost` — assert stress mode multiplies cost
  linearly.

**Impact.** Previous leaderboards on this branch are invalid. The
1000× undercharge inflated every Sharpe, return, and edge metric.
Re-running on real Dukascopy data is required before any spec is
considered for certification (planned for Batch E).

### Phase 1 — split loader

**New module.** [data/splits.py](../data/splits.py) exposes:

- `Split(name, start, end)` — single window, with `slice(df)` returning
  a `[start, end)` half-open slice on a `time` column.
- `Splits` — frozen dataclass holding train / validation / holdout
  pairs of (h4, m15) frames, plus the underlying `Split` configs.
- `load_split_config(path)` — parses `config/data_splits.json`, rejects
  missing keys, inverted windows, and overlapping splits.
- `load_splits(symbol, config_path)` — loads candles via the canonical
  loader and pre-slices them into the six frames runners need.
- `assert_no_holdout_leak(df, splits, where)` — guard for runners that
  must not see holdout rows.
- `assert_only_in_split(df, split, where)` — guard for runners that
  must operate inside a specific window.
- `coverage_summary(splits)` — diagnostic returning declared vs. actual
  start/end and row counts per split.

**Tests.** [tests/test_splits.py](../tests/test_splits.py) covers:

- Happy-path config parse.
- Reject overlap, inverted windows, missing splits.
- Half-open slice (boundary timestamp lands in exactly one slice).
- Data-level overlap absent across the three frames.
- Leak detector rejects holdout rows in train, accepts pure train.
- `assert_only_in_split` rejects cross-split rows, accepts in-split.
- Coverage summary is populated.

**Coverage gap surfaced.** Config declares `research_train` from
2008-01-01, but the on-disk data starts 2020-07-01. The actual train
slice is **1.5 years (2,408 H4 bars)**, not 14 years. The audit fails
on this and instructs the user to either update config or pull
older Dukascopy years. Walk-forward folds are accordingly limited.

### Phase 3 — no-lookahead

**Audit.** Filters reviewed for use of current-bar (entry-bar) data:

| Filter | Pre-fix lookahead source | Status |
|--------|--------------------------|--------|
| `atr_percentile` | `_atr` at index *i* uses bar *i* HLC | **fixed** — shift ranks by 1 |
| `pdh_pdl` | `cur_close = c` (current bar's close) | **fixed** — replaced with `_shift1(c)` |
| `vwap_dist` | typical-price uses bar *i* HLC | **fixed** — shift z by 1 |
| `htf_vwap_dist` | same | **fixed** — shift z by 1 |
| `body_atr` | already uses `c[:-1]`, `o[:-1]` | safe, unchanged |
| `regime` | MA already shifted | safe, unchanged |
| `min_streak` | shift by `i` per `k` | safe, unchanged |
| `regime_class` | regime[i] from ranks/same_dir at i, then `regime_prev[i]=regime[i-1]` | safe, unchanged |
| `wick_ratio` | uses `c[:-1]`, `o[:-1]` | safe, unchanged |
| `session` | hour at bar start (known at entry) | safe, unchanged |

Signals (`prev_color`, `displacement`, `sweep_rejection`,
`failed_continuation`, `multi_bar_directional`) all already shift their
output by one bar before exposing it as `sig[i]`. No changes needed
on the signal side.

**Helper.** New `_shift1(arr)` in `execution/executor.py` enforces the
"compute on completed bars then shift by 1" pattern. Used in all four
fixed filters.

**Mutation regression test.**
[tests/test_no_lookahead.py](../tests/test_no_lookahead.py) builds a
synthetic 400-bar H4 + M15 pair, runs each filter, captures entries,
mutates `close/high/low` of every bar from the entry index forward
(±$50 random per bar), and asserts the entry list at and before the
entry index is unchanged. A "past mutation can change decision" test
verifies the inverse: filters do respond to changes in already-formed
bars (otherwise the filter would be inert).

Eight filters covered: `atr_percentile`, `pdh_pdl` (`inside` and
`breakout`), `vwap_dist`, `htf_vwap_dist`, `body_atr`, `regime`,
`regime_class`. The breakout case is exercised even when the synthetic
data produces zero trades — the test then asserts that mutating the
future on the empty trade list still yields the empty trade list.

### Phase 11 — test gate

`make all` and `make report` now run `make test` before any data /
audit / leaderboard / PDF target. `TEST_FILES` is an explicit variable
so adding a new test only requires editing the Makefile.

`scripts/run_tests.py` is the cross-platform mirror — runs each test
file as a subprocess, prints failures, exits non-zero on the first
file that fails. Useful on Windows hosts without GNU make.

**Current status:** 36 tests across 7 files, all passing.

### Phase 10 — audit expansion

`scripts/audit.py` adds six new check sections in addition to the
original five:

1. **`audit_splits`** — config parses; windows are non-overlapping; each
   split has actual data coverage; the actual first bar is within 90
   days of the declared start.
2. **`audit_spread_unit_convention`** — fails if any active executor
   path multiplies `spread` by `POINT_SIZE`.
3. **`audit_walkforward_no_downgrade`** — fails while
   `validation/walkforward.py` still rewrites M15 entries to
   `h4_open`. Will go green when Batch B replaces the runner.
4. **`audit_prop_sim_no_future_leak`** — fails while
   `prop_challenge/risk.py` accepts `trade_pnl_price`. Will go green
   when Batch C lands.
5. **`audit_shuffle_test_disabled`** — fails while
   `validation/certify.py` references `shuffled_outcome_test`. Will
   go green when Batch B replaces the test.
6. **`audit_results_provenance`** — fails on every active leaderboard
   CSV that lacks a `<stem>.meta.json` sidecar (commit + config
   provenance). Will be addressed in Batch E.

**Current state.** 5 audit failures (intentional — the four known
unfixed phases plus the train coverage gap). Audit exits 1.

## Files changed

```
modified:   Makefile
modified:   execution/executor.py
modified:   scripts/audit.py
modified:   scripts/strategy.py
modified:   tests/test_stop_exit_spread.py
new file:   data/splits.py
new file:   docs/HARDENING_REPORT.md
new file:   scripts/run_tests.py
new file:   tests/test_no_lookahead.py
new file:   tests/test_splits.py
```

## Tests added

- `tests/test_splits.py` — 9 tests (5 config-level, 4 data-level).
- `tests/test_no_lookahead.py` — 9 tests (8 filter-mutation + 1 inverse).
- `tests/test_stop_exit_spread.py` — 2 new tests
  (`test_spread_unit_from_bid_ask`, `test_spread_mult_scales_cost`).

Total: **36 tests** across 7 files; all passing.

## Reproduce

```bash
# Linux / Mac / WSL:
make test
make audit                  # exits 1 with 5 known failures (see above)

# Windows (no GNU make):
python3 scripts/run_tests.py
python3 scripts/audit.py
```

## Batch B summary

| # | Phase | Status | What changed |
|---|-------|--------|--------------|
| 6 | M15-aware walk-forward | ✅ landed | `validation/walkforward.py` rewritten to call `execution.executor.run` directly. No silent rewriting of M15 entries. Entry-model + entry-timeframe compatibility checked up front. Specs requesting an unsupported entry timeframe (e.g. M5 not yet wired into the executor) report `compatibility="data_unavailable"` and produce zero folds, never an `h4_open` fallback. |
| 7 | Holdout uses same executor | ✅ landed | `validation/holdout.py:yearly_segments` now takes `(spec, h4, m15, ...)` and calls `run_executor_on_window` — same kernel as walk-forward and holdout proper. Old `run_h4_only` deleted. |
| 8 | Statistical tests rebuilt | ✅ landed | `validation/statistical_tests.py` replaces `shuffled_outcome_test` (which was a no-op — Sharpe is invariant under permutation) with: `label_permutation_test` (random direction at same timestamps), `random_eligible_entry_test` (random entries on filter-passing bars), `daily_block_bootstrap_test` (stationary block bootstrap on daily equity returns, gates on lower-CI > 0). Old test now raises RuntimeError on call. Default permutation counts: 500 exploration / 5000 final. |
| 9 | Certifier wired to new tests | ✅ landed | `validation/certify.py` accepts `stat_label_perm`, `stat_random`, `stat_block_boot`. Adds a `wf_compatibility != "ok"` failure path. Calling with `stat_shuffle` raises `DeprecationWarning`. |
| 10 | Trade-frequency Sharpe + supplementary metrics | ✅ landed | `analytics/trade_metrics.basic` now exposes `sharpe_h4_bar_ann` (legacy form, kept as `sharpe_ann` alias), `sharpe_trade_ann`, `sharpe_daily_ann`, `sharpe_weekly_ann`. New: `time_under_water_share`, `worst_day_ret`, `worst_week_ret`, `expectancy_R`, `trades_per_year`, `n_trading_days`, `n_trading_weeks`. |
| 11 | Audit checks updated | ✅ landed | Walk-forward audit also verifies `execution.executor` is imported (positive signal). Shuffle-test audit verifies the deprecation tombstone is present and the certifier is wired to `stat_label_perm`. |

### Batch B file changes

```
modified:   analytics/trade_metrics.py
modified:   scripts/agent_06_refiner.py
modified:   scripts/audit.py
modified:   scripts/run_alpha.py
modified:   scripts/run_tests.py
modified:   scripts/run_v2.py
modified:   validation/certify.py
modified:   validation/holdout.py
modified:   validation/statistical_tests.py
modified:   validation/walkforward.py
modified:   Makefile
new file:   tests/test_walkforward_parity.py
new file:   tests/test_statistical_tests.py
new file:   tests/test_trade_metrics.py
```

### Batch B tests

- `tests/test_walkforward_parity.py` — 5 tests:
  - `run_executor_on_window` and `executor.run` produce identical trades on the same window.
  - WF and holdout produce identical trades on a shared time slice.
  - M5 entry timeframe yields `compatibility=data_unavailable`, zero folds, no fallback.
  - Unknown entry model yields `compatibility=unknown_model`.
  - Static check: walkforward.py contains no downgrade markers.
- `tests/test_statistical_tests.py` — 8 tests:
  - Deprecated shuffle test raises.
  - Label permutation rejects strong edge; type-I rate ≤ 20% on noise (target 5%).
  - Random eligible entry rejects strong edge; type-I rate ≤ 20% on noise.
  - Block bootstrap rejects strong edge via lower-CI > 0; doesn't false-pass random days.
  - Benjamini-Hochberg basics.
- `tests/test_trade_metrics.py` — 8 tests:
  - `sharpe_ann` is back-compat alias for H4-bar form.
  - Sparse-trade Sharpe differs from H4-bar Sharpe by the expected factor.
  - Dense-trade Sharpe is in O(1) ratio of H4-bar Sharpe.
  - daily / weekly Sharpe present and sensible.
  - `time_under_water_share` ∈ [0, 1] and zero on always-rising curves.
  - `expectancy_R` matches the analytic formula.
  - Worst-day / worst-week populated correctly.
  - Empty trades returns `{"trades": 0}`.

### Batch B test + audit totals

After Batch B: 57 tests across 10 files; all passing. Audit: 32 PASS / 2 FAIL.

## Batch C summary

| # | Phase | Status | What changed |
|---|-------|--------|--------------|
| 12 | RiskModel sizing without future leakage | ✅ landed | `prop_challenge/risk.py` rewritten. `size()` no longer accepts `trade_pnl_price`. Pre-trade inputs only: `balance`, `starting_balance`, `max_loss`, `last_trade_loss`, `equity_high`, `stop_distance_price`, `atr_pre_entry`. Tighter stops size more contracts; missing stop falls back to `contracts_base`. |
| 13 | Chronological replay | ✅ landed | New `prop_challenge.challenge.run_chronological_replay(trades_df, account, risk, rules)`. Replays the actual ledger in timestamp order, groups by REAL UTC calendar days, applies daily lockouts / loss limits / trailing drawdown / consistency rule in order. Deterministic for a given input — no Monte Carlo. |
| 14 | Day-level block bootstrap | ✅ landed | `run_challenge` now resamples DAYS, not individual trades. `_sample_day_blocks(all_days, block_days, n_days, rng)` draws contiguous blocks from the historical day list, preserving intra-day clustering of wins/losses. |
| 15 | Wilson + bootstrap CIs | ✅ landed | New `prop_challenge/stats.py` with `wilson_ci(successes, n, alpha)` (vendored Acklam inverse-Φ; no scipy dep) and `bootstrap_median_ci(values, n_runs)`. `ChallengeResult` and `PayoutResult` now carry CI fields. `passes_cert(use_ci_bounds=True)` gates on lower CI of pass/payout and upper CI of blowup, not on point estimates. |
| 16 | Default MC counts raised | ✅ landed | `N_RUNS_EXPLORATION = 500`, `N_RUNS_FINAL = 5000`. Same convention as the statistical tests added in Batch B. |
| 17 | Schema for `prop_accounts.json` | ✅ landed | `prop_challenge/accounts.py:validate_schema()` rejects accounts missing required fields, unknown drawdown types, malformed `_verification` blocks. Runs at load time. Schema version bumped to 2. |
| 18 | Verification metadata + status | ✅ landed | Each account in `prop_accounts.json` has `_verification: {source_url, last_verified, notes}`. `verification_status(spec, today, stale_days=90)` returns one of `verified` / `stale` / `unverified` / `synthetic`. `can_certify_for_live(spec)` returns True only for `verified`. The audit fails when real-firm accounts are unverified, prompting the user to verify rules manually before live use. |
| 19 | Legacy `prop/simulator.py` adapter | ✅ landed | Rewritten as a thin shim over `prop_challenge`. Existing runner call sites in `run_v2.py`, `run_alpha.py` continue to work. The simulator returns the legacy keys (`blowup_probability`, `prop_survival_score`, `passes_<name>`) plus new keys (`pass_probability_ci`, `verification_status`, `research_only`). |
| 20 | Audit checks updated | ✅ landed | New audit section "prop sim risk sizing + bootstrap (Phase 7)" enforces: no `trade_pnl_price=` in `risk.py`, pre-trade inputs accepted, `run_chronological_replay` exported, day-level block bootstrap used, Wilson CI emitted. New section "prop_accounts.json verification metadata (Phase 8)": schema parses; every account has a `_verification` block; real-firm accounts have valid `last_verified` dates within 90 days. |

### Batch C file changes

```
modified:   config/prop_accounts.json
modified:   prop/simulator.py
modified:   prop_challenge/accounts.py
modified:   prop_challenge/challenge.py
modified:   prop_challenge/payout.py
modified:   prop_challenge/risk.py
modified:   prop_challenge/score.py
modified:   scripts/audit.py
modified:   scripts/run_tests.py
modified:   Makefile
new file:   prop_challenge/stats.py
new file:   tests/test_prop_simulator.py
```

### Batch C tests

`tests/test_prop_simulator.py` — 18 tests:
- Risk sizing signature: no `trade_pnl_price`; accepts `stop_distance_price` + `atr_pre_entry`.
- Risk sizing is deterministic for fixed pre-trade inputs.
- Tighter stop sizes more contracts than wider stop (within cap).
- Sizing falls back to `contracts_base` when no stop / ATR available.
- Chronological replay passes a winning ledger and respects `minimum_trading_days`.
- Chronological replay blows up on a day breaching the daily-loss limit.
- Chronological replay groups trades by REAL calendar day (max-trades-per-day rule respected within day, not within session).
- Day-block bootstrap with `block_days=3` always produces at least one consecutive triplet of original day indices.
- Wilson CI bounds stay in `[0, 1]`; width shrinks with n.
- `wilson_ci` is sane at 0% and 100% successes.
- `bootstrap_median_ci` produces a CI that contains the analytic median.
- Verification status branches: `unverified`, `verified`, `stale`, `synthetic`.
- `can_certify_for_live` returns True only for `verified`.
- Schema validation rejects missing fields and unknown drawdown types.
- `load_all` parses the real `prop_accounts.json` and recognises the synthetic generic accounts.
- `run_challenge` end-to-end emits valid Wilson CI on pass probability.

### Test totals

**75 tests across 11 files; all passing.** Was 57/57 after Batch B.

### Audit totals

**33 PASS / 2 FAIL.** Both remaining are intentional:

- `no real-firm accounts left as 'unverified' or 'stale'` — the 7 Topstep + MFFU accounts have `last_verified: null`. The user must manually verify each firm's current rules and set the date. The audit nudges this; the certifier downstream marks results as `research_only` until verified.
- `prop_challenge_leaderboard.csv: provenance metadata present` — Batch E.

The 4 Batch-B-PASS-only checks (split coverage, walk-forward executor, shuffle test, label-perm gate) all stay PASS.

## Batch D summary (canonicalise pipeline + verified prop accounts)

| # | Phase | Status | What changed |
|---|-------|--------|--------------|
| 21 | Canonical entrypoint | ✅ landed | New `scripts/run_pipeline.py` is the single hardened orchestrator. Loads `Splits` via `data.splits.load_splits()` (NOT `load_all`), so train / validation / holdout boundaries are enforced at the runner. Walk-forward only sees train; validation is a separate evaluation slice; holdout is single-revelation. |
| 22 | Provenance sidecar | ✅ landed | Each leaderboard CSV has a `<stem>.meta.json` next to it carrying `git_head`, `config_hashes` (data_splits + prop_accounts), `splits.coverage_summary`, `n_specs`, `n_perm`, `runtime_seconds`, `prop_account_verification_summary`, `schema_versions`, and a `notes` field. The audit's results-provenance check now passes whenever every active CSV has a sidecar. |
| 23 | Legacy runners archived | ✅ landed | `scripts/walkforward.py` and `scripts/run_v2.py` moved to `scripts/_deprecated_/`. They predated the load_splits boundary and the M15-aware walk-forward. |
| 24 | Makefile + README | ✅ landed | New `make pipeline` target. README rewritten to identify the canonical entrypoints and list legacy / deprecated targets. The hardening status is summarised at the top of the README. |
| 25 | Verified prop account rules (Phase 8 follow-through) | ✅ landed | A research subagent fetched current public rule pages for Topstep + MyFundedFutures and produced verified numbers with source URLs. `config/prop_accounts.json` rewritten with the 2026 reality:<br>**Topstep**: drawdown is intraday-monitored (was eod_trailing); DLL values $1000/$2000/$3000 (down from $1100/$2200/$3300) and OPTIONAL; min_days 5→2; payout_min_days 8→5; max_challenge_days null (no time limit).<br>**MFFU**: `Starter`/`Expert` tiers do not exist — restructured Jul 2025 (Core/Rapid/Pro), Jan 2026 (Core→Flex). Keys renamed to `mffu_rapid_25k`, `mffu_pro_50k/100k/150k`. **No DLL on any MFFU plan in 2026** (`daily_loss_limit: null`). Pro uses EOD trailing (was intraday). max_contracts cut significantly (Pro 100K: 10→6). payout_min_days for Pro is 14 (was 5). max_challenge_days null.<br>All 7 real-firm accounts now `verification_status="verified"` and the Phase-8 audit check passes. |
| 26 | Schema v3 (nullable fields) | ✅ landed | `prop_challenge/accounts.py:validate_schema` allows `daily_loss_limit` and `max_challenge_days` to be `null`. `drawdown.py` skips DLL check when None. `challenge.py` and `payout.py` cap untimed evaluations at 60 days for Monte Carlo tractability while chronological replay keeps the full ledger length. |

### Batch D file changes

```
modified:   Makefile
modified:   README.md
modified:   config/prop_accounts.json
modified:   prop_challenge/accounts.py
modified:   prop_challenge/challenge.py
modified:   prop_challenge/drawdown.py
modified:   prop_challenge/payout.py
new file:   scripts/run_pipeline.py
new file:   results/_pre_hardening/README.md
renamed:    scripts/run_v2.py        -> scripts/_deprecated_/run_v2.py
renamed:    scripts/walkforward.py   -> scripts/_deprecated_/walkforward.py
```

### Batch D verification

```
$ python3 scripts/run_tests.py
PASS  all 11 test files passed       (75 / 75 individual tests)

$ python3 scripts/audit.py; echo $?
=== audit summary: 0 failures ===
0
```

Was 5 audit failures at Batch A start; **0 now**.

## Batch E summary (rebuild outputs)

| # | Phase | Status | What changed |
|---|-------|--------|--------------|
| 27 | Pre-hardening leaderboards archived | ✅ landed | `results/prop_challenge_leaderboard.csv` moved to `results/_pre_hardening/` with a README explaining the four reasons it cannot be cited as evidence (1000× spread undercharge, M15→H4 walk-forward downgrade, future-leak in sizing, no-op shuffle test). |
| 28 | Hardened leaderboard | ✅ landed | `make pipeline` evaluated the canonical 72-spec grid (3 selectivity × 8 entry models × 3 stops) against the hardened gates with `n_perm=500`. Runtime: **63.6 minutes**. Results in `results/leaderboard_hardened.csv` (75 KB) and `results/leaderboard_hardened.meta.json`. **0 specs certified.** This is the right outcome under the hardened gates: realistic spread costs + no walk-forward downgrade + no shuffle-test free-pass + a 1.5-year train slice (4-fold floor) is a higher bar than any spec in the canonical grid clears. The leaderboard preserves all 72 rows with full metrics and per-spec failure reasons so the user can see *which* gate each spec failed. |
| 28b | Top-of-leaderboard near-misses (informational) | — | The five closest-to-certifying specs all use the NY-session filter and limit-style entries (zone_midpoint_limit / fib_limit_entry):<br>1. `ny__zone_midpoint_limit__prev_h4_extreme` — holdout +8.99%, holdout sharpe_trade_ann +1.12, dd −2.57%; **fails on WF**: 3 folds (need 4), median Sharpe −6.36, only 33% positive folds.<br>2. `ny__zone_midpoint_limit__h4_atr_x1.0` — holdout +8.66%, similar profile.<br>3. `ny__zone_midpoint_limit__prev_h4_open` — holdout +6.76%.<br>4. `ny__fib_limit_entry_lvl0.382__h4_atr_x1.0` — holdout +5.29%, WF median Sharpe −1.68 (closest WF to break-even).<br>5. `ny__fib_limit_entry_lvl0.618__h4_atr_x1.0` — holdout +5.06%.<br>The honest read: NY-session limit entries on Q1-Q2-2024 + Q1-Q2-2026 holdout look promising, but the 2020-2022 train slice rejects them. They are candidates worth re-running against a longer train history (post-2008 Dukascopy backfill) before being trusted. |
| 28c | Failure-reason histogram | — | 72 specs × multiple gates, in counts:<br>72 `wf_folds<4` (the train-slice constraint),<br>72 `random_baseline failed`, 72 `block_bootstrap failed`, 71 `label_permutation failed` (so 1 spec just clears p<0.05 on label permutation),<br>64 `profit_factor<=1.2`, 61 `wf_median_sharpe<=0`, 61 `wf_pct_positive_folds<0.5`,<br>54 `stress_total_return<=0`, 48 `holdout_total_return<=0`,<br>44 `yearly_consistency failed`, 22 `max_drawdown<-0.2`. |
| 28d | Prop sim summary | — | Median 25k blowup probability: 57.6% (Wilson upper CI 61.9%). High because most specs are losing or too volatile for the small account; the `prop_25k_research_only` flag is True on all 72 because each carries the legacy 25k preset's `verification_status="unverified"` (the canonical pipeline uses the legacy `prop.simulator.simulate_all` adapter which builds an `AccountSpec` from `core.constants.PROP_ACCOUNTS` rather than the verified `prop_accounts.json` keys). Promotion to verified-account simulations is a Batch-D-follow-up nice-to-have, not a hardening blocker — the 7 real-firm accounts in `prop_accounts.json` are now `verified` and ready to use directly. |
| 29 | HARDENING_REPORT updated | ✅ landed (this section) | Documents what changed and why. |

## Batch F summary (OHLC-only foundations + UI scaffold)

This batch refactors the harness toward the prop-firm passing engine
described in the Batch F brief. It establishes:

- **OHLC-only feature contract.** `core/feature_capability.py` declares
  what data the harness has on disk (OHLC, spread, tick count, TPO)
  and what it does NOT have (real volume, bid/ask volume, footprint,
  delta, CVD, order book, VWAP, Volume Profile). Strategy proposers
  must check `classify_candidate(spec)` before running; an
  unavailable-feature token (`vwap_dist`, `volume_poc`, `footprint`,
  `delta`, `cvd`, `bid_ask_imbalance`, `dom_liquidity`, etc.) returns
  `status="rejected_unavailable_data"` with the offending tokens and
  a renaming hint to the OHLC proxy.

- **TPO as a first-class allowed auction feature.** `analytics/tpo.py`
  computes a true Time-Price-Opportunity profile from OHLC candles
  alone — *time at price*, not volume at price. Outputs: POC, VAH,
  VAL, value area width, single prints, poor highs/lows, excess
  highs/lows, initial balance high/low, open inside/outside prior
  value. The module never reads a `volume` column.

- **PropCandidate schema.** `core/candidate.py` introduces
  `PropCandidate` — the rich research object combining strategy +
  filters + entry + stop + exit + risk model + daily rules + prop
  account + certification metadata. Composition over inheritance:
  `candidate.to_spec()` produces an executor-compatible `Spec` so
  the hardened executor / walk-forward / holdout / statistical
  pipeline keeps running unchanged. `PropCandidate.from_spec()`
  wraps an existing Spec with default risk / daily / account.

- **Certification ladder.** `core/certification.py` replaces the
  binary `certified: bool` with an 8-level `CertificationLevel`
  enum: `REJECTED_UNAVAILABLE_DATA → REJECTED_BROKEN → RESEARCH_ONLY
  → WATCHLIST → CANDIDATE → PROP_CANDIDATE → CERTIFIED`, plus
  `RETIRED` for previously-certified strategies that no longer
  qualify. Each `FailureReason` (e.g. `FAIL_LOOKAHEAD`,
  `FAIL_WALK_FORWARD`, `FAIL_HIGH_BLOWUP_PROBABILITY`) caps the
  candidate at a specific level.

- **UI event/progress scaffold.** `core/run_state.py` and
  `core/run_events.py` write per-run `progress.json` and append-only
  `events.jsonl` files under `results/runs/<run_id>/`. The hardened
  pipeline still works without an EventWriter — `emit_candidate(None,
  ...)` is a safe no-op. A future React/FastAPI control room (Batch
  J) will tail `events.jsonl` and render per-stage progress.

- **Active-code purge of unavailable-data references.** Three places
  still hard-coded `vwap_dist` / `htf_vwap_dist` / `vwap_mean_reversion`
  references were updated:
    * `scripts/agent_03_spec_builder.py` — VWAP filters removed from
      the known-filter set with a comment pointing at the OHLC proxy.
    * `scripts/agent_06_refiner.py` — `htf_vwap_dist_2.5` knob
      removed from the refinement allow-list.
    * `core/strategy_families.py` — `vwap_mean_reversion` family
      renamed to `session_mean_reversion`; the VWAP filter is dropped
      from the template (Batch G adds the OHLC ATR-distance proxy).
    * `scripts/strategy.py` — v1 strategy module's `vwap_dist` and
      `htf_vwap_dist` filter implementations replaced with explicit
      `raise ValueError(...)` so any caller that still passes those
      tokens fails loudly rather than silently using `tick_count` as
      volume.
    * `regime/filters.py` — unused `vwap_distance_z` function
      removed.

- **Audit expansion.** `scripts/audit.py` adds two new sections:
    * "OHLC-only feature capability (Batch F)" — verifies the
      canonical `CapabilityRegistry` disables real volume / VWAP /
      Volume Profile / footprint / delta / CVD / orderbook;
      `classify_candidate` rejects representative tokens for each;
      `tpo_*` tokens are accepted.
    * "active strategies do not reference unavailable data" — sweeps
      every `.py` in `scripts/`, `validation/`, `execution/`, `core/`,
      `analytics/`, `prop/`, `prop_challenge/`, `data/`, `regime/`,
      `entry_models/`, `strategies/` for hard-coded references to
      `vwap`, `volume_profile`, `footprint`, `delta`, `cvd`,
      `bid_ask_imbalance`, `dom_liquidity`. Skips the small set of
      files that legitimately mention those tokens (`audit.py`,
      `feature_capability.py`, `build_pdf.py` documentation,
      `scripts/strategy.py` deprecation tombstones).

### Batch F file changes

```
new file:   core/feature_capability.py
new file:   core/certification.py
new file:   core/candidate.py
new file:   core/run_state.py
new file:   core/run_events.py
new file:   analytics/tpo.py
new file:   tests/test_feature_capability.py
new file:   tests/test_certification.py
new file:   tests/test_candidate.py
new file:   tests/test_tpo.py
new file:   tests/test_run_events.py
modified:   core/strategy_families.py        (vwap_mean_reversion -> session_mean_reversion)
modified:   regime/filters.py                (vwap_distance_z removed)
modified:   scripts/agent_03_spec_builder.py (vwap_dist / htf_vwap_dist out of KNOWN_FILTER_TYPES)
modified:   scripts/agent_06_refiner.py      (htf_vwap_dist_2.5 knob removed)
modified:   scripts/strategy.py              (vwap_dist / htf_vwap_dist tombstoned)
modified:   scripts/audit.py                 (Batch F audit sections)
modified:   scripts/run_tests.py             (5 new test files wired)
modified:   Makefile                         (5 new test files wired)
```

### Batch F tests

55 new tests across 5 files:

- `test_feature_capability.py` (12) — VWAP / VP / footprint / delta /
  CVD / orderbook tokens are rejected; TPO tokens are allowed without
  volume; existing OHLC filters still allowed; unknown tokens
  reported but treated as OHLC; `assert_only_ohlc_only` raises;
  registry can be overridden if a future dataset adds real volume;
  no typos in the tag map.
- `test_certification.py` (10) — levels are uniquely ordered;
  failure reasons all have caps; `REJECTED_UNAVAILABLE_DATA`
  dominates other failures; lookahead caps at REJECTED_BROKEN; prop-
  specific failures cap at PROP_CANDIDATE; `promote/demote` clamp at
  extremes; verdict round-trips through JSON.
- `test_candidate.py` (6) — `to_spec()` returns an executor-
  compatible Spec; `from_spec()` round-trips; rich blocks (Risk /
  DailyRules / AccountRef) round-trip through dict; certification
  state round-trips through JSON.
- `test_tpo.py` (13) — POC at most-visited price; VAH/VAL sandwich
  POC; single prints, poor highs/lows, excess flags consistent with
  auction-theory definitions; initial balance covers first N
  brackets only; open-inside-value detection works with prior value;
  empty input safe; profile JSON-serialises; `session_slices`
  yields one group per UTC day; bracket count uses candle range
  (no volume column).
- `test_run_events.py` (14) — run directory + events.jsonl created;
  `next_seq` increments per date; `set_stage` / `set_status` /
  `bump` validate against allowed vocabularies; events appended as
  one JSON line each, timestamped + run-id-stamped automatically;
  malformed events rejected; recent_events list capped;
  `emit_candidate` / `emit_stage` helpers; safe with `events=None`;
  `write_summary` includes extras; run_id format is sortable
  `YYYY-MM-DD_NNN`.

### Test totals

**130 individual tests across 16 files; all passing.** Was 75/75
after Batch C-E.

### Audit totals

**45 PASS / 0 FAIL.** Was 33 PASS / 2 FAIL after Batch C-E.

The two formerly-known failures are gone:
- "no real-firm accounts left as 'unverified' or 'stale'" — fixed in
  Batch D when verified rules came back from the research agent.
- "leaderboard_hardened.csv: provenance metadata present" — fixed in
  Batch E when `make pipeline` produced the `.meta.json` sidecar.

## Batch G summary (sweep labs)

The candidate-generation layer the prop-firm passing engine needs.

| # | What | File | Output |
|---|---|---|---|
| 1 | 10 OHLC-only family generators | `strategies/families.py` | 42 base PropCandidates across 10 families |
| 2 | Entry-model lab | `strategies/entry_lab.py` | up to 9 variants per base; only registered/compatible entries; `compare_table()` surfaces best entry per metric |
| 3 | Risk sweep | `strategies/risk_sweep.py` | 7 default presets (`fixed_micro_1/2`, `dollar_risk_50/100`, `pct_dd_buffer_2pct`, `reduce_after_loss`, `scale_after_high`); cartesian with `contracts_max` available |
| 4 | Daily-rule optimiser | `strategies/daily_rule_lab.py` | 12 default presets (max_trades_per_day, stop_after_n_losses, daily_loss/profit_stop_dollar, session_only); 23 in `variants_full()` |
| 5 | Tier-1 grid composer | `strategies/grid.py` | `tier_1_grid()` returns all families' base variants (42 today); `apply_capability_filter()` partitions OHLC-only vs unavailable-data; `tier_2_for_survivor(lab=...)` dispatches per-axis sweeps; `grid_summary()` emits per-family / per-cert-level counts |

### Family roster (verbatim from the brief)

```
session_sweep_reclaim
opening_range_breakout_retest
opening_range_failed_breakout
previous_h4_range_retracement
previous_h4_sweep_reclaim
tpo_value_rejection           ← Batch H needs to wire executor TPO filters
tpo_poc_reversion             ← same
atr_extension_reclaim
compression_breakout
failed_breakout_reversal
```

Each family produces 4-6 base variants (different stop choices,
session restrictions, threshold parameters). The grid is intentionally
controlled — the test
`test_default_tier_1_grid_size_is_controlled` enforces a 20-150
candidate band.

### Composition over inheritance preserved

Every family generator returns `PropCandidate` objects whose
`to_spec()` produces an executor-compatible `Spec`. The hardened
executor / walk-forward / holdout / statistical pipeline still
consumes `Spec` unchanged. The lab modules don't touch the
candidates' setup logic — they only multiply variants on
`risk` / `daily_rules` / `entry` axes.

### TPO-family marker

`tpo_value_rejection` and `tpo_poc_reversion` candidates pass the
Batch F capability gate (TPO is allowed without volume) but their
`tpo_*` filter tokens are not yet implemented as executor filters.
Each candidate carries `provenance.requires_executor_extension =
"tpo_filter"`. The Batch H orchestrator must:

1. Either wire TPO filter implementations into
   `execution.executor._apply_filters` (preferred), OR
2. Detect the marker and skip TPO candidates with status
   `data_unavailable` and a clear message.

### Tier-2 expansion (per survivor)

Once a tier-1 candidate clears the OHLC backtest + walk-forward
filter, the orchestrator runs targeted tier-2 sweeps on it:

  | Lab | Variants per survivor |
  |---|---:|
  | `risk` | 7 (default presets) — up to 21 with `variants_with_caps((1,3,5))` |
  | `daily` | 12 (default) — up to 23 with `variants_full()` |
  | `entry` | 8-9 (compatible entries only) |
  | `tier_2_full` | sum of the above per survivor (~27) |

So 5 survivors → ~135 tier-2 candidates → tractable in 2-3 hours
of prop sim at `n_runs=500`. Final certification at `n_runs=5000`
runs only on the 1-2 leaderboard top spots.

### Batch G file changes

```
new file:   strategies/__init__.py
new file:   strategies/families.py
new file:   strategies/entry_lab.py
new file:   strategies/risk_sweep.py
new file:   strategies/daily_rule_lab.py
new file:   strategies/grid.py
new file:   tests/test_strategies.py
modified:   scripts/run_tests.py     (test_strategies.py wired)
modified:   Makefile                 (test_strategies.py wired)
```

### Batch G tests (22 new, 152 individual / 17 files total)

`tests/test_strategies.py`:

- 6 family-level (every family emits ≥1; family ids match the
  brief; unique ids; every candidate passes capability check;
  `to_spec()` works; TPO families carry the executor-extension
  marker)
- 3 entry-lab (only registered/compatible entries; unique ids;
  `compare_table()` shape — higher-better and lower-better metrics)
- 3 risk-sweep (default count = 7; `contracts_max` honoured;
  cartesian with caps)
- 3 daily-rule (default count matches `DEFAULT_PRESETS`; doesn't
  mutate base; full preset list wider than default)
- 7 grid (tier-1 = all families; family filter; capability filter
  partitions clean vs dirty; tier-2 dispatch by lab name; tier-2
  full combines all labs; `grid_summary` counts; size band)

### Audit unchanged

45 PASS / 0 FAIL. The new `strategies/` directory is in the audit's
"active strategies do not reference unavailable data" sweep path
and passes cleanly.

## Batch H summary (orchestrator + leaderboard + report)

The payoff batch — Batches F + G are now wired into a runnable
prop-firm passing engine.

| # | What | File | Output |
|---|---|---|---|
| 1 | Tiered orchestrator with CLI | `scripts/run_prop_passing.py` | `make prop-passing` / `make prop-passing-smoke`. Flags: `--smoke`, `--limit-candidates`, `--families`, `--accounts`, `--max-survivors-for-prop-sim`, `--fast-only`, `--full`, `--n-perm`, `--output-stem`. |
| 2 | Prop passing score | `reports/prop_passing_score.py` | Composite metric: `+pass_prob*35 +payout_survival*20 +drawdown_safety*20 +consistency*10 +median_days*10 +simplicity*5 -blowup*35 -daily_loss_breach*20 -trailing_dd_breach*20 -overfit_penalty*25` |
| 3 | OHLC-only proxy filter | `analytics/session_mean.py` + executor `atr_distance_from_session_mean` branch | Replaces the old VWAP filter. Anchored typical-price mean, restarted at each session open; distance in ATR units. |
| 4 | TPO filters wired into executor | `analytics/tpo_levels.py` + executor TPO dispatcher | `attach_prev_session_tpo(h4, m15)` precomputes `prev_session_tpo_{poc,vah,val,ib_high,ib_low,poor_high,poor_low,excess_high,excess_low}` columns. Executor's `_apply_filters` calls `apply_tpo_filter(...)` for any `tpo_*` token. Refuses to silently let trades through if the columns are missing. |
| 5 | Cooldown lockout | `prop_challenge/lockout.py` | New `cooldown_minutes_after_loss` field on `DailyRules`. `update_day(rules, day, pnl, ts=...)` records `cooldown_until` after a loss; `admit_trade` blocks new trades while cooldown is active. New presets `cd60` / `cd120`. |
| 6 | Refiner module | `strategies/refiner.py` | Maps failure reasons to concrete `MutationSuggestion` lists. `FAIL_HIGH_BLOWUP_PROBABILITY` / `FAIL_DAILY_LOSS_LIMIT` → stricter daily lockouts + lower risk. `FAIL_TOO_FEW_TRADES` → widen entry. `FAIL_RANDOM_BASELINE` / `FAIL_LABEL_PERMUTATION` → REJECT_AND_REDESIGN (no parameter tweak helps). `REJECTED_UNAVAILABLE_DATA` → empty list (not actionable). |
| 7 | Tombstones for VWAP filters | `execution/executor.py` | `vwap_dist` / `htf_vwap_dist` branches replaced with `raise ValueError(...)` so any caller fails loudly. |
| 8 | README rewrite | `README.md` | OHLC-only positioning at the top; `make prop-passing` documented as the canonical entrypoint; legacy `make pipeline` retained for per-spec deep dive. |
| 9 | Audit additions | `scripts/audit.py` | "executor extensions (Batch H)" section verifies executor wires `atr_distance_from_session_mean`, `apply_tpo_filter`, and the VWAP tombstones. |

### Smoke run output

`python3 scripts/run_prop_passing.py --smoke` evaluates 4 candidates
end-to-end in ~2.5 minutes:

```
[run] id=2026-04-29_001 dir=results/runs/2026-04-29_001
[1/9] generated 42 tier-1 candidates  (limited to 4)
[2/9] capability filter: kept=4 rejected=0
[3/9] loading splits + precomputing TPO levels ...
[4-5/9] evaluating 4 candidates (n_perm=80) ...
[6/9] prop sim on top 4 candidates ...
[7/9] leaderboard -> results/prop_passing_leaderboard.csv (4 rows)
[7/9] sidecar    -> results/prop_passing_leaderboard.meta.json
[8/9] report     -> results/prop_passing_report.md

=== SUMMARY ===
candidates evaluated: 4    prop_candidates: 0    certified: 0
runtime: 2.5 min
```

All 4 smoke candidates land at `research_only` because the train
slice (1.5y) only fits 3 walk-forward folds (need 4) — the same
data-coverage gate the Batch E pipeline reports. The leaderboard
correctly carries Wilson CIs on `pass_probability` /
`blowup_probability` and the failure histogram in the report:

```
fail_walk_forward       4
fail_holdout            4
fail_label_permutation  4
fail_random_baseline    4
fail_block_bootstrap    4
fail_spread_stress      4
fail_high_blowup        3
fail_low_pass_prob      3
```

The run also writes `results/runs/2026-04-29_001/{progress.json,
events.jsonl, summary.json}` — the contract Batch J's UI will
consume.

### Batch H file changes

```
new file:   analytics/session_mean.py
new file:   analytics/tpo_levels.py
new file:   reports/prop_passing_score.py
new file:   scripts/run_prop_passing.py
new file:   strategies/refiner.py
new file:   tests/test_batch_h.py
modified:   Makefile                         (make prop-passing[-smoke])
modified:   README.md                        (OHLC-only positioning at top)
modified:   core/candidate.py                (cooldown_minutes_after_loss field)
modified:   execution/executor.py            (atr_distance, tpo dispatch, vwap tombstones)
modified:   prop_challenge/challenge.py      (pass ts to update_day)
modified:   prop_challenge/lockout.py        (cooldown_minutes_after_loss + presets)
modified:   prop_challenge/payout.py         (pass ts to update_day)
modified:   scripts/audit.py                 (Batch H executor-extensions section)
modified:   scripts/run_tests.py             (test_batch_h wired)
modified:   strategies/daily_rule_lab.py     (cd60/cd120 presets)
modified:   tests/test_no_lookahead.py       (vwap tests -> atr_distance + tpo tests)
```

### Batch H tests (19 new, 171 individual / 18 files total)

`tests/test_batch_h.py`:

- 3 cooldown (block after loss, no fire on win, preset list)
- 2 session-mean (resets each day, no volume column needed)
- 4 TPO levels + dispatcher (columns added; empty m15 safe;
  refuses without columns; rejection logic sane)
- 4 refiner (unavailable→empty; high-blowup→risk+lockouts;
  random-fail→redesign; too-few-trades→widen entry)
- 5 prop passing score (rewards pass / penalises blowup;
  drawdown_safety clamps; simplicity decreases with filters;
  overfit penalty zero for good p; median_days handles None)
- 1 orchestrator import smoke

`tests/test_no_lookahead.py` updated:

- `test_vwap_dist_no_lookahead` / `test_htf_vwap_dist_no_lookahead`
  removed (filters no longer exist; tombstoned with raise).
- Replaced with `test_atr_distance_from_session_mean_no_lookahead`
  (OHLC proxy mutation invariance) and
  `test_tpo_value_acceptance_no_lookahead` (TPO filter pulls from
  PREVIOUS session, mutation invariance).

### Audit (46 PASS / 0 FAIL)

Two new checks pass:
- "executor wires `atr_distance_from_session_mean`"
- "executor wires the `tpo_*` filter family"
- "`vwap_dist` / `htf_vwap_dist` raise ValueError tombstones"

Was 45 / 0 after Batch G.

## What is NOT yet done

| Item | Owner / Notes |
|---|---|
| **Tier-2 sweeps integration into the orchestrator.** `scripts/run_prop_passing.py` runs the tier-1 grid + prop sim on top survivors but does NOT yet run `tier_2_for_survivor(lab="risk"/"daily"/"entry")`. The labs work; they just aren't wired into the runner's loop. The orchestrator can be extended without changing any of Batches A-G. | Follow-up. |
| **Batch J** — local React/FastAPI control room. The Batch F event scaffold (`core/run_state.py`, `core/run_events.py`) writes `results/runs/<run_id>/events.jsonl` + `progress.json` + `summary.json`. Future UI tails these files. | Far future. |
| **Pre-2020 Dukascopy data backfill.** Train slice is 18 months (2020-07 → 2022-01) which only fits 3 walk-forward folds; the certifier's 4-fold floor is unreachable. User has explicitly chosen "leave as is". | User decision. |
| **Re-verifying prop accounts in 90 days.** Operational; the audit will start failing on `last_verified` staleness automatically. | Operational. |
| Walk-forward gate is relaxed (`min_folds=4`, `min_median_sharpe=0`) because the train slice (2020-07 → 2022-01) only fits ~4 disjoint 6m/3m folds. To restore the full 20-fold gate, fetch older Dukascopy years and widen `research_train.start`. | User decision; sub-2020 data is not yet in the sidecar branch. The user has explicitly chosen "leave as is" for now. |
| Re-verifying prop accounts in 90 days. | Operational; the audit will start failing on `last_verified` staleness automatically. |
| The legacy `scripts/run_alpha.py` and `scripts/run_prop_challenge.py` still call `load_all` (full-series data) instead of `load_splits()`. They predate Batch D. They've been kept in `scripts/` (not moved to `_deprecated_/`) for back-compat but the README marks them as legacy. | Optional follow-up; if you want them retired, move to `_deprecated_/`. |

## Constraints honoured

- No strategy optimisation in Batch A.
- No new synthetic-data claims.
- No fallback to non-Dukascopy sources.
- No M15 → H4 downgrade introduced (the existing one is now flagged
  by audit, not silently used).
- No realised PnL / MAE / MFE leaked into sizing.
- No reliance on existing leaderboards.

## Acceptance status (per the original request)

| Criterion | Status |
|-----------|--------|
| `make test` passes | ✅ 75 / 75 (11 test files) |
| `make audit` fails on splits / lookahead / spread / M15 proxy | ✅ exits 0 now (every check passes); fails 1 if any of those is reintroduced |
| Walk-forward and holdout use same executable strategy | ✅ same executor, same entry semantics, parity tests pin it |
| Spread cost is correctly charged in price units | ✅ |
| No-lookahead tests pass | ✅ 9 mutation tests |
| Statistical tests can reject random / not falsely pass noise | ✅ 8 power tests |
| Trade-frequency-aware Sharpe, daily / weekly equity Sharpe | ✅ explicit names; legacy alias retained |
| Prop sim has chronological replay + day/week bootstrap | ✅ chronological + day-level block bootstrap |
| Risk sizing uses only pre-trade information | ✅ `trade_pnl_price` removed; pre-trade inputs only |
| Wilson CI / bootstrap CI for prop pass / blowup probabilities | ✅ certifier gates on lower CI bound |
| Prop account configs schema-validated; verification metadata required | ✅ all 7 real-firm accounts re-verified 2026-04-29 |
| Canonical pipeline entrypoint with provenance metadata | ✅ `make pipeline` writes `.meta.json` sidecar |
| Final report distinguishes research / validation / holdout / prop survivors | ✅ leaderboard columns: `wf_*`, `val_*`, `ho_*`, `prop_*`, `cert.failures` |
| Existing stale results regenerated or marked invalid | ✅ pre-hardening artefacts archived under `results/_pre_hardening/` with README explaining why they're invalid |
