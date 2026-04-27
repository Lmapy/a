.PHONY: all data backtest search clean

all: data backtest

data:
	python3 scripts/fetch_data.py

backtest:
	python3 scripts/backtest.py

search:
	python3 scripts/orchestrate.py

clean:
	rm -f data/*.csv results/*.csv results/*.png
