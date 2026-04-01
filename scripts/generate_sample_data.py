#!/usr/bin/env python3
"""Generate realistic synthetic ES futures intraday data for backtesting.

Uses a mean-reverting random walk with realistic volatility, volume patterns,
and session structure to simulate ES futures 5-minute bars.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_es_data(n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate n_days of realistic ES 5-minute bar data.

    Simulates:
    - Session hours: 8:30 AM - 3:00 PM CT (78 five-minute bars per day)
    - Realistic price movement (~0.1% per bar std dev for ES)
    - Volume profile: U-shaped (high at open/close, low midday)
    - Occasional trending and ranging days
    - Opening gaps between sessions
    """
    rng = np.random.RandomState(seed)
    bars_per_day = 78  # 6.5 hours * 12 bars/hour

    all_rows = []
    price = 5200.0  # starting ES price

    for day in range(n_days):
        date = pd.Timestamp("2025-01-06") + pd.Timedelta(days=day)
        # Skip weekends
        if date.weekday() >= 5:
            continue

        # Overnight gap: small random move
        gap = rng.normal(0, 5.0)  # ~5 point overnight gap std
        price += gap

        # Day type: 30% trending, 70% ranging
        is_trending = rng.random() < 0.30
        if is_trending:
            trend_direction = rng.choice([-1, 1])
            trend_strength = rng.uniform(0.02, 0.08)  # points per bar drift
        else:
            trend_direction = 0
            trend_strength = 0

        # Daily volatility regime
        daily_vol = rng.uniform(0.08, 0.25)  # % per bar

        session_start = date.replace(hour=8, minute=30)

        for bar_idx in range(bars_per_day):
            bar_time = session_start + pd.Timedelta(minutes=5 * bar_idx)

            # Volume profile: U-shaped
            t = bar_idx / bars_per_day
            volume_profile = 2.5 - 3.0 * t * (1 - t)  # high at edges
            base_volume = rng.poisson(int(15000 * max(volume_profile, 0.5)))

            # Price movement
            drift = trend_direction * trend_strength
            noise = rng.normal(0, daily_vol)
            mean_reversion = -0.005 * (price - 5200)  # weak pull back to 5200

            bar_return = drift + noise + mean_reversion
            bar_open = price

            # Simulate intrabar price action
            intra_moves = rng.normal(0, daily_vol / 3, 4)
            intra_prices = bar_open + np.cumsum(intra_moves)
            bar_close = bar_open + bar_return

            bar_high = max(bar_open, bar_close, *intra_prices) + abs(rng.normal(0, daily_vol * 0.3))
            bar_low = min(bar_open, bar_close, *intra_prices) - abs(rng.normal(0, daily_vol * 0.3))

            # Snap to tick size (0.25)
            bar_open = round(bar_open * 4) / 4
            bar_high = round(bar_high * 4) / 4
            bar_low = round(bar_low * 4) / 4
            bar_close = round(bar_close * 4) / 4

            # Ensure OHLC consistency
            bar_high = max(bar_high, bar_open, bar_close)
            bar_low = min(bar_low, bar_open, bar_close)

            all_rows.append({
                "datetime": bar_time,
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": max(base_volume, 100),
            })

            price = bar_close

    df = pd.DataFrame(all_rows)
    df.set_index("datetime", inplace=True)
    return df


def main():
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("Generating 60 days of synthetic ES 5-minute data...")
    df = generate_es_data(n_days=90, seed=42)  # ~60 trading days

    filepath = output_dir / "ES_5min.csv"
    df.to_csv(filepath)
    print(f"Saved {len(df)} bars to {filepath}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Avg daily volume: {df['volume'].mean():,.0f}")


if __name__ == "__main__":
    main()
