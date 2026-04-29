.PHONY: all pipeline prop-passing prop-passing-smoke data pull-data audit \
        alpha prop report test pdf clean ui ui-stdlib \
        cbr-scalp cbr-scalp-smoke cbr-scalp-sweep \
        cbr-walkforward cbr-monte-carlo cbr-props cbr-charts \
        cbr-london cbr-ny \
        fib backtest search skeptic v2

# CANONICAL TARGETS (post-hardening, Batches A-D)
#
#   make test       -> run the harness test suite (gates everything else)
#   make audit      -> structural audit of the harness
#   make pull-data  -> pull Dukascopy candles from the sidecar branch
#   make pipeline   -> canonical hardened pipeline:
#                       splits -> walk-forward (train) -> validation
#                       -> holdout -> prop sim with chronological replay
#                       -> leaderboard_hardened.csv + .meta.json
#   make all        -> test, audit, pipeline, pdf
#   make report     -> alias of `all`
#   make pdf        -> regenerate the audit PDF
#
# LEGACY TARGETS (pre-hardening; kept for back-compat). New work should
# go through `make pipeline`. See `scripts/_deprecated_/` for runners
# that have been moved out of the canonical path entirely.
#
#   make alpha      -> scripts/run_alpha.py
#   make prop       -> scripts/run_prop_challenge.py
#   make data       -> scripts/fetch_dukascopy.py (network-bound)


all: test audit prop-passing pdf

# Batch H canonical entrypoint -- prop-firm passing engine.
# Tiered runner; ranks by prop_passing_score; produces
# prop_passing_leaderboard.csv + .meta.json + prop_passing_report.md
# plus per-run progress/events files under results/runs/.
prop-passing:
	python3 scripts/run_prop_passing.py

prop-passing-smoke:
	python3 scripts/run_prop_passing.py --smoke

# Hardened legacy 72-spec pipeline (Batch E). Kept for the per-spec
# deep-dive use case; does NOT rank by prop_passing_score.
pipeline:
	python3 scripts/run_pipeline.py

# OFFICIAL DATA: Dukascopy (single source). Old broker fetcher is
# deprecated; see data/_deprecated_/. Default range is the period
# covered by config/data_splits.json.
data:
	python3 scripts/fetch_dukascopy.py --symbol XAUUSD --start 2008-01-01 --end 2026-04-29

pull-data:
	python3 scripts/pull_sidecar.py

audit:
	python3 scripts/audit.py

# legacy alpha-search runner (kept for back-compat)
alpha:
	python3 scripts/run_alpha.py

prop:
	python3 scripts/run_prop_challenge.py

report: test audit pipeline pdf

# `make test` runs every test file as a script. Each file's __main__
# block prints PASS lines and exits non-zero on the first failure.
TEST_FILES := \
	tests/test_executor.py \
	tests/test_registry.py \
	tests/test_compatibility.py \
	tests/test_stop_exit_spread.py \
	tests/test_validator.py \
	tests/test_splits.py \
	tests/test_no_lookahead.py \
	tests/test_walkforward_parity.py \
	tests/test_statistical_tests.py \
	tests/test_trade_metrics.py \
	tests/test_prop_simulator.py \
	tests/test_feature_capability.py \
	tests/test_certification.py \
	tests/test_candidate.py \
	tests/test_tpo.py \
	tests/test_run_events.py \
	tests/test_strategies.py \
	tests/test_batch_h.py \
	tests/test_ui.py \
	tests/test_cbr_scalp.py \
	tests/test_cbr_extensions.py

test:
	@set -e; for f in $(TEST_FILES); do \
		echo "=== $$f ==="; \
		python3 $$f || exit 1; \
	done

pdf:
	python3 scripts/build_pdf.py

# Batch J -- local research control room.
# Reads results/runs/<run_id>/ + results/prop_passing_leaderboard.csv
# and serves a small JSON API + a vanilla-JS frontend at localhost:8765.
ui:
	python3 ui/server.py

ui-stdlib:
	python3 ui/server.py --no-fastapi

# Batch K -- CBR-style gold scalp module.
# Mechanical approximation of a public TomTrades / CBR scalping
# model; runs separately from the prop-passing / pipeline paths.
cbr-scalp:
	python3 scripts/run_cbr_backtest.py

cbr-scalp-smoke:
	python3 scripts/run_cbr_backtest.py --start 2024-01-01 --limit-bars 100000 --output-stem cbr_smoke

cbr-scalp-sweep:
	python3 scripts/run_cbr_sweep.py --start 2024-01-01 --limit-bars 200000

# Batch K2 extensions
cbr-walkforward:
	python3 scripts/run_cbr_walkforward.py

cbr-monte-carlo:
	python3 scripts/run_cbr_montecarlo.py

cbr-props:
	python3 scripts/run_cbr_props.py

cbr-charts:
	python3 scripts/run_cbr_charts.py

cbr-london:
	python3 scripts/run_cbr_backtest.py --config configs/cbr_gold_scalp_london.yaml

cbr-ny:
	python3 scripts/run_cbr_backtest.py --config configs/cbr_gold_scalp_ny.yaml

clean:
	rm -f results/*.csv results/*.png results/*.txt results/*.json results/*.meta.json
