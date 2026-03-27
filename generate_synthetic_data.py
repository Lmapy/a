"""Generate synthetic ES futures bar data for testing/demo purposes.

Uses a geometric Brownian motion model with:
- Realistic ES price level (~5000)
- Daily volatility ~0.8% (ES historical ~15% annualised)
- Slight upward drift
- Realistic OHLCV structure (high > close > open > low, etc.)
"""
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_bars(
    n_bars: int = 500,
    start_price: float = 4800.0,
    daily_vol: float = 0.008,
    drift: float = 0.0003,
    seed: int = 42,
    start_date: datetime = datetime(2023, 1, 3, 9, 30),
    skip_weekends: bool = True,
) -> list[dict]:
    random.seed(seed)

    def randn() -> float:
        # Box-Muller
        u1 = random.random() + 1e-10
        u2 = random.random()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    bars = []
    price = start_price
    dt = start_date

    while len(bars) < n_bars:
        # Skip weekends
        if skip_weekends and dt.weekday() >= 5:
            dt += timedelta(days=1)
            continue

        ret = drift + daily_vol * randn()
        close = round(price * (1 + ret) * 4) / 4  # round to ES tick (0.25)
        open_ = round(price * (1 + daily_vol * 0.3 * randn()) * 4) / 4
        high = round(max(open_, close) * (1 + abs(daily_vol * 0.5 * randn())) * 4) / 4
        low  = round(min(open_, close) * (1 - abs(daily_vol * 0.5 * randn())) * 4) / 4
        volume = int(random.gauss(800_000, 150_000))

        bars.append({
            "datetime": dt.isoformat(),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": max(volume, 100_000),
        })

        price = close
        dt += timedelta(days=1)

    return bars


def save(bars: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(bars)
    print(f"Saved {len(bars)} bars → {path}")


if __name__ == "__main__":
    # Training data: 2 years
    train = generate_bars(n_bars=500, start_price=4800.0, seed=42)
    save(train, "/tmp/ES_train.csv")
    print(f"  Price range: {min(b['close'] for b in train):.2f} – {max(b['close'] for b in train):.2f}")

    # Test data: 6 months (different seed → out-of-sample)
    test = generate_bars(n_bars=130, start_price=train[-1]["close"], seed=99, drift=0.0004)
    save(test, "/tmp/ES_test.csv")
    print(f"  Price range: {min(b['close'] for b in test):.2f} – {max(b['close'] for b in test):.2f}")
