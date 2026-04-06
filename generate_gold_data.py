"""
Generate realistic synthetic gold (XAU/USD) 1-minute candle data for 5 years.
Uses realistic parameters:
- Gold daily volatility ~1% (annualized ~15-20%)
- Trending behavior with mean reversion
- Session-based volume and volatility patterns (Asian, London, NY)
- Realistic spread between OHLC
- Incorporates regime changes (trending/ranging)
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_gold_1min(years=5, start_price=1800.0, seed=42):
    """
    Generate realistic 1-minute gold candle data.

    Gold characteristics:
    - Annualized volatility: ~15-18%
    - 1-min volatility: ~0.01-0.03%
    - Strong trend persistence
    - Session-based patterns
    - Occasional spikes (news events)
    """
    rng = np.random.RandomState(seed)

    # Trading hours: ~23 hours/day (gold trades nearly 24h), 5 days/week
    # ~252 trading days/year, ~1380 minutes/day (23h * 60)
    bars_per_day = 1380
    trading_days_per_year = 252
    total_days = int(years * trading_days_per_year)
    total_bars = total_days * bars_per_day

    print(f"Generating {total_bars:,} 1-minute bars ({total_days} days, {years} years)")

    # Generate dates (skip weekends)
    start_date = pd.Timestamp('2021-04-06')
    dates = []
    current_date = start_date
    days_generated = 0

    while days_generated < total_days:
        if current_date.weekday() < 5:  # Monday-Friday
            for minute in range(bars_per_day):
                hour = minute // 60
                mins = minute % 60
                dates.append(current_date + pd.Timedelta(hours=hour, minutes=mins))
            days_generated += 1
        current_date += pd.Timedelta(days=1)

    dates = dates[:total_bars]

    # --- Price Generation ---
    # Use a regime-switching model for realistic gold behavior

    # Base volatility per minute (annualized ~17%)
    annual_vol = 0.17
    daily_vol = annual_vol / np.sqrt(252)
    minute_vol = daily_vol / np.sqrt(bars_per_day)

    # Regime model: trending (0) vs ranging (1)
    regime = np.zeros(total_bars, dtype=int)
    current_regime = 0
    regime_duration = 0

    for i in range(total_bars):
        regime_duration += 1
        # Average regime lasts 5-20 days
        if rng.random() < 1.0 / (bars_per_day * rng.randint(5, 20)):
            current_regime = 1 - current_regime
            regime_duration = 0
        regime[i] = current_regime

    # Generate returns with regime-dependent drift and volatility
    returns = np.zeros(total_bars)
    trend_direction = 1  # 1 = up, -1 = down
    trend_strength = 0.0

    for i in range(total_bars):
        # Time-of-day volatility multiplier
        hour = dates[i].hour if i < len(dates) else 12
        if 0 <= hour < 8:  # Asian session: lower vol
            vol_mult = 0.6
        elif 8 <= hour < 12:  # London open: high vol
            vol_mult = 1.4
        elif 12 <= hour < 16:  # London-NY overlap: highest vol
            vol_mult = 1.6
        elif 16 <= hour < 20:  # NY afternoon: moderate
            vol_mult = 1.0
        else:  # Late session: low vol
            vol_mult = 0.5

        # Regime-dependent behavior
        if regime[i] == 0:  # Trending
            # Persistent drift
            if rng.random() < 0.001:  # Occasional trend reversal
                trend_direction *= -1
            trend_strength = 0.00002 * trend_direction
            drift = trend_strength
            vol = minute_vol * vol_mult * 1.1
        else:  # Ranging
            drift = 0
            vol = minute_vol * vol_mult * 0.8
            # Mean reversion component
            if i > 100:
                ma = np.mean(returns[max(0, i-100):i])
                drift = -ma * 0.01

        # News spike simulation (rare high-vol events)
        if rng.random() < 0.0001:  # ~1 spike per 10000 bars
            vol *= rng.uniform(5, 15)

        returns[i] = drift + vol * rng.standard_normal()

    # Convert returns to prices
    close_prices = np.zeros(total_bars)
    close_prices[0] = start_price

    for i in range(1, total_bars):
        close_prices[i] = close_prices[i-1] * (1 + returns[i])

    # Add realistic uptrend to gold (gold went from ~1800 to ~3000+ over 2021-2026)
    # Add a slow drift up
    trend = np.linspace(0, 0.65, total_bars)  # ~65% total gain over 5 years
    close_prices = close_prices * (1 + trend)

    # Generate OHLC from close prices
    open_prices = np.zeros(total_bars)
    high_prices = np.zeros(total_bars)
    low_prices = np.zeros(total_bars)
    volumes = np.zeros(total_bars)

    open_prices[0] = close_prices[0]

    for i in range(total_bars):
        if i > 0:
            # Open = previous close + small gap
            gap = rng.normal(0, minute_vol * close_prices[i] * 0.3)
            open_prices[i] = close_prices[i-1] + gap

        # High and Low
        bar_range = abs(close_prices[i] - open_prices[i])
        extra_high = abs(rng.normal(0, minute_vol * close_prices[i] * 0.5))
        extra_low = abs(rng.normal(0, minute_vol * close_prices[i] * 0.5))

        high_prices[i] = max(open_prices[i], close_prices[i]) + extra_high
        low_prices[i] = min(open_prices[i], close_prices[i]) - extra_low

        # Volume (session-dependent)
        hour = dates[i].hour if i < len(dates) else 12
        if 0 <= hour < 8:
            base_vol = 50
        elif 8 <= hour < 16:
            base_vol = 200
        else:
            base_vol = 100
        volumes[i] = max(1, int(base_vol * rng.lognormal(0, 0.5)))

    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices[:len(dates)],
        'High': high_prices[:len(dates)],
        'Low': low_prices[:len(dates)],
        'Close': close_prices[:len(dates)],
        'Volume': volumes[:len(dates)].astype(int),
    }, index=pd.DatetimeIndex(dates))

    return df


def resample_to_timeframes(df_1m):
    """Resample 1-min data to multiple timeframes."""
    results = {}

    # 1-minute (last 7 days worth)
    bars_7d = 7 * 1380
    results['1m_recent'] = df_1m.iloc[-bars_7d:]

    # 5-minute (last 60 days)
    bars_60d = 60 * 1380
    df_5m = df_1m.iloc[-bars_60d:].resample('5min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    results['5m_60d'] = df_5m

    # 15-minute (last 60 days)
    df_15m = df_1m.iloc[-bars_60d:].resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    results['15m_60d'] = df_15m

    # 1-hour (last 2 years)
    bars_2y = 2 * 252 * 1380
    df_1h = df_1m.iloc[-bars_2y:].resample('1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    results['1h_2y'] = df_1h

    # Daily (full 5 years)
    df_1d = df_1m.resample('1D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    results['1d_5y'] = df_1d

    return results


def main():
    print("Generating 5 years of synthetic gold 1-minute data...")
    df_1m = generate_gold_1min(years=5, start_price=1800.0, seed=42)

    print(f"\nGenerated {len(df_1m):,} 1-minute bars")
    print(f"Date range: {df_1m.index[0]} to {df_1m.index[-1]}")
    print(f"Price range: ${df_1m['Close'].min():.2f} to ${df_1m['Close'].max():.2f}")

    print("\nResampling to multiple timeframes...")
    datasets = resample_to_timeframes(df_1m)

    for name, df in datasets.items():
        path = os.path.join(DATA_DIR, f"gold_{name}.csv")
        df.to_csv(path)
        print(f"  {name}: {len(df):,} bars -> {path}")

    print("\nDone! All data saved.")
    return datasets


if __name__ == "__main__":
    main()
