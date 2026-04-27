.PHONY: all data backtest clean

all: data backtest

data:
	python3 scripts/fetch_data.py

backtest:
	python3 scripts/backtest.py

clean:
	rm -f data/*.csv results/*.csv results/*.png
