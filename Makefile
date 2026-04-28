.PHONY: all data audit fib backtest search skeptic v2 alpha prop report test pdf clean

all: data audit alpha prop pdf

# OFFICIAL DATA: Dukascopy (single source). Old broker fetcher is deprecated;
# see data/_deprecated_/. Default range is the period covered by config/data_splits.json.
data:
	python3 scripts/fetch_dukascopy.py --symbol XAUUSD --start 2008-01-01 --end 2026-04-29

audit:
	python3 scripts/audit.py

fib:
	python3 scripts/fib_analysis.py

backtest:
	python3 scripts/backtest.py

search:
	python3 scripts/orchestrate.py

skeptic:
	python3 scripts/skeptic.py

v2:
	python3 scripts/run_v2.py

alpha:
	python3 scripts/run_alpha.py

prop:
	python3 scripts/run_prop_challenge.py

report: data audit alpha prop pdf

test:
	python3 tests/test_executor.py
	python3 tests/test_registry.py
	python3 tests/test_compatibility.py
	python3 tests/test_stop_exit_spread.py

pdf:
	python3 scripts/build_pdf.py

clean:
	rm -f results/*.csv results/*.png results/*.txt results/*.json
