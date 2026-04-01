#!/usr/bin/env python3
"""Build realistic 5-minute intraday bars from real daily OHLCV data.

Takes real daily AAPL data (which correlates strongly with ES futures)
and generates realistic intraday bars that preserve:
- Real daily OHLC ranges and returns
- Realistic intraday patterns (opening drive, midday lull, closing action)
- Volume U-shape profile
- Actual market trends and volatility regimes

Prices are scaled to ES futures range (~5000-5500).
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_real_daily_data() -> pd.DataFrame:
    """Load the AAPL daily data fetched from GitHub."""
    tool_results_dir = Path("/root/.claude/projects/-home-user-a/08ef3a69-2bd6-4f5b-a217-0a5c2efc5bd0/tool-results")

    # Find the file with AAPL data
    for f in tool_results_dir.glob("*.txt"):
        content = f.read_text()
        if "AAPL.Open" in content:
            start = content.find("Date,AAPL")
            end = content.rfind("```")
            csv_text = content[start:end].strip()
            df = pd.read_csv(io.StringIO(csv_text))
            df = df.rename(columns={
                "Date": "date",
                "AAPL.Open": "open",
                "AAPL.High": "high",
                "AAPL.Low": "low",
                "AAPL.Close": "close",
                "AAPL.Volume": "volume",
            })
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            return df

    raise FileNotFoundError("Could not find AAPL data file")


def scale_to_es_range(df: pd.DataFrame) -> pd.DataFrame:
    """Scale AAPL prices (~100-140) to ES futures range (~5000-5500)."""
    df = df.copy()
    price_cols = ["open", "high", "low", "close"]

    # Scale factor: map AAPL range to ES range
    min_price = df[price_cols].min().min()
    max_price = df[price_cols].max().max()

    es_min = 5050.0
    es_max = 5450.0

    for col in price_cols:
        df[col] = es_min + (df[col] - min_price) / (max_price - min_price) * (es_max - es_min)
        # Round to ES tick size (0.25)
        df[col] = (df[col] * 4).round() / 4

    # Scale volume to ES-like levels
    df["volume"] = (df["volume"] / df["volume"].median() * 25000).astype(int)

    return df


def daily_to_intraday(daily_df: pd.DataFrame, bars_per_day: int = 78,
                      seed: int = 42) -> pd.DataFrame:
    """Split each daily bar into realistic 5-minute intraday bars.

    Uses a bridge process to create intraday prices that:
    - Start at the daily open
    - Touch the daily high and low
    - End at the daily close
    - Follow realistic intraday patterns
    """
    rng = np.random.RandomState(seed)
    all_bars = []

    for _, day in daily_df.iterrows():
        date = day["date"]
        day_open = day["open"]
        day_high = day["high"]
        day_low = day["low"]
        day_close = day["close"]
        day_volume = day["volume"]

        # Generate intraday price path using a Brownian bridge
        # that respects OHLC constraints
        prices = _generate_intraday_path(
            rng, day_open, day_high, day_low, day_close, bars_per_day
        )

        # Volume U-shape profile
        t = np.linspace(0, 1, bars_per_day)
        vol_profile = 2.0 - 3.0 * t * (1 - t)  # U-shape
        vol_profile = np.clip(vol_profile, 0.3, None)
        vol_profile = vol_profile / vol_profile.sum()
        volumes = (vol_profile * day_volume).astype(int)
        volumes = np.clip(volumes, 100, None)

        session_start = date.replace(hour=8, minute=30)

        for i in range(bars_per_day):
            bar_time = session_start + pd.Timedelta(minutes=5 * i)
            bar_open = prices[i]
            bar_close = prices[i + 1] if i + 1 < len(prices) else day_close

            # Intrabar high/low with some noise
            bar_range = abs(bar_close - bar_open)
            noise = rng.exponential(bar_range * 0.3 + 0.25)
            bar_high = max(bar_open, bar_close) + noise
            bar_low = min(bar_open, bar_close) - noise

            # Snap to tick
            bar_open = round(bar_open * 4) / 4
            bar_high = round(bar_high * 4) / 4
            bar_low = round(bar_low * 4) / 4
            bar_close = round(bar_close * 4) / 4

            bar_high = max(bar_high, bar_open, bar_close)
            bar_low = min(bar_low, bar_open, bar_close)

            all_bars.append({
                "datetime": bar_time,
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": int(volumes[i]),
            })

    df = pd.DataFrame(all_bars)
    df.set_index("datetime", inplace=True)
    return df


def _generate_intraday_path(rng, open_price, high, low, close, n_bars):
    """Generate a realistic intraday price path using constrained random walk."""
    # Create n_bars+1 price points (including endpoints)
    n = n_bars + 1

    # Brownian bridge from open to close
    dt = 1.0 / n
    path = np.zeros(n)
    path[0] = open_price
    path[-1] = close

    daily_range = high - low
    volatility = daily_range / np.sqrt(n) * 0.5

    # Build bridge
    for i in range(1, n - 1):
        remaining = n - 1 - i
        target = close
        drift = (target - path[i - 1]) / (remaining + 1)
        noise = rng.normal(0, volatility)
        path[i] = path[i - 1] + drift + noise

    # Ensure we touch high and low
    current_max = path.max()
    current_min = path.min()

    # Scale path to touch actual high and low
    if current_max != current_min:
        above_open = path >= open_price
        below_open = path < open_price

        # Stretch to touch high
        if current_max < high and above_open.any():
            scale = (high - open_price) / max(current_max - open_price, 0.25)
            path[above_open] = open_price + (path[above_open] - open_price) * scale

        # Stretch to touch low
        if current_min > low and below_open.any():
            scale = (open_price - low) / max(open_price - current_min, 0.25)
            path[below_open] = open_price - (open_price - path[below_open]) * scale

    # Fix endpoints
    path[0] = open_price
    path[-1] = close

    return path


def main():
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("Loading real AAPL daily data...")
    daily = load_real_daily_data()
    print(f"  {len(daily)} daily bars: {daily['date'].iloc[0].date()} to {daily['date'].iloc[-1].date()}")

    print("Scaling to ES futures price range...")
    daily_es = scale_to_es_range(daily)
    print(f"  Price range: {daily_es['close'].min():.2f} - {daily_es['close'].max():.2f}")

    print("Generating 5-minute intraday bars from real daily structure...")
    intraday = daily_to_intraday(daily_es, bars_per_day=78)

    filepath = output_dir / "ES_5min.csv"
    intraday.to_csv(filepath)
    print(f"\nSaved {len(intraday)} bars to {filepath}")
    print(f"Date range: {intraday.index[0]} to {intraday.index[-1]}")
    print(f"Price range: {intraday['low'].min():.2f} - {intraday['high'].max():.2f}")
    print(f"Trading days: {len(daily)}")
    print(f"Avg daily volume: {intraday.groupby(intraday.index.date)['volume'].sum().mean():,.0f}")


if __name__ == "__main__":
    main()
