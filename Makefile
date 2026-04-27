.PHONY: all data audit backtest search clean

all: data audit backtest

data:
	python3 scripts/fetch_data.py

audit:
	python3 scripts/audit.py

backtest:
	python3 scripts/backtest.py

search:
	python3 scripts/orchestrate.py

clean:
	rm -f data/*.csv results/*.csv results/*.png results/*.txt
