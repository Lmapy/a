.PHONY: all data audit fib backtest search skeptic v2 alpha report test pdf clean

all: data audit backtest

data:
	python3 scripts/fetch_data.py

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

report: data audit alpha pdf

test:
	python3 tests/test_validator.py
	python3 tests/test_executor.py
	python3 tests/test_registry.py
	python3 tests/test_compatibility.py
	python3 tests/test_stop_exit_spread.py

pdf:
	python3 scripts/build_pdf.py

clean:
	rm -f data/*.csv results/*.csv results/*.png results/*.txt
