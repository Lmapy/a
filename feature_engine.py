"""
Feature engineering for gold trading ML model.
Computes technical indicators and price action features without external TA libraries
to avoid dependency issues.
"""

import pandas as pd
import numpy as np


def sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(series, period=20, std_dev=2):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return wr


def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = atr(high, low, close, 1)  # true range (1 period = raw TR)
    # Smooth with EMA
    atr_smooth = ema(tr, period)
    plus_di = 100 * ema(plus_dm, period) / atr_smooth.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = ema(dx, period)
    return adx_val, plus_di, minus_di


def compute_features(df):
    """
    Compute all features from OHLCV data.
    Expects columns: Open, High, Low, Close, Volume
    Returns DataFrame with all features added.
    """
    df = df.copy()

    # Ensure we have the right columns
    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    v = df.get('Volume', pd.Series(0, index=df.index))

    # --- Price Action Features ---
    df['body'] = c - o
    df['body_pct'] = df['body'] / o * 100
    df['upper_wick'] = h - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - l
    df['range'] = h - l
    df['range_pct'] = df['range'] / c * 100

    # Returns at various lookbacks
    for lb in [1, 3, 5, 10, 20, 50]:
        df[f'ret_{lb}'] = c.pct_change(lb) * 100

    # --- Moving Averages ---
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{p}'] = sma(c, p)
        df[f'ema_{p}'] = ema(c, p)
        df[f'close_vs_sma_{p}'] = (c - df[f'sma_{p}']) / df[f'sma_{p}'] * 100

    # MA crossover signals
    df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['sma_10_50_cross'] = (df['sma_10'] > df['sma_50']).astype(int)
    df['sma_20_100_cross'] = (df['sma_20'] > df['sma_100']).astype(int)
    df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)

    # --- RSI ---
    for p in [7, 14, 21]:
        df[f'rsi_{p}'] = rsi(c, p)

    # --- Bollinger Bands ---
    bb_upper, bb_mid, bb_lower = bollinger_bands(c, 20, 2)
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100
    df['bb_position'] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # --- ATR ---
    for p in [7, 14, 21]:
        df[f'atr_{p}'] = atr(h, l, c, p)
        df[f'atr_{p}_pct'] = df[f'atr_{p}'] / c * 100

    # --- MACD ---
    macd_line, signal_line, hist = macd(c)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['macd_cross'] = (macd_line > signal_line).astype(int)

    # --- Stochastic ---
    k, d = stochastic(h, l, c)
    df['stoch_k'] = k
    df['stoch_d'] = d
    df['stoch_cross'] = (k > d).astype(int)

    # --- Williams %R ---
    df['williams_r'] = williams_r(h, l, c)

    # --- ADX ---
    adx_val, plus_di, minus_di = adx(h, l, c)
    df['adx'] = adx_val
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['di_cross'] = (plus_di > minus_di).astype(int)

    # --- Volatility Features ---
    df['volatility_10'] = c.pct_change().rolling(10).std() * 100
    df['volatility_20'] = c.pct_change().rolling(20).std() * 100
    df['volatility_50'] = c.pct_change().rolling(50).std() * 100

    # --- Volume Features (if available) ---
    if v.sum() > 0:
        df['vol_sma_10'] = sma(v, 10)
        df['vol_sma_20'] = sma(v, 20)
        df['vol_ratio'] = v / df['vol_sma_20'].replace(0, np.nan)

    # --- Session/Time Features ---
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        # Session flags (approximate)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['london_ny_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

    # --- Higher Timeframe Features (multi-timeframe analysis) ---
    # Resample to get higher TF context
    for tf_bars in [5, 15, 60]:
        htf_close = c.rolling(tf_bars).mean()
        htf_high = h.rolling(tf_bars).max()
        htf_low = l.rolling(tf_bars).min()
        df[f'htf_{tf_bars}_trend'] = (c - htf_close) / htf_close * 100
        df[f'htf_{tf_bars}_range_pos'] = (c - htf_low) / (htf_high - htf_low).replace(0, np.nan)

    # --- Support/Resistance Proximity ---
    for lb in [20, 50, 100]:
        rolling_high = h.rolling(lb).max()
        rolling_low = l.rolling(lb).min()
        df[f'dist_to_high_{lb}'] = (rolling_high - c) / c * 100
        df[f'dist_to_low_{lb}'] = (c - rolling_low) / c * 100

    # --- Momentum Features ---
    df['momentum_5'] = c - c.shift(5)
    df['momentum_10'] = c - c.shift(10)
    df['momentum_20'] = c - c.shift(20)
    df['roc_5'] = c.pct_change(5) * 100
    df['roc_10'] = c.pct_change(10) * 100
    df['roc_20'] = c.pct_change(20) * 100

    # --- Pattern Features ---
    # Consecutive up/down bars
    df['up_bar'] = (c > o).astype(int)
    df['consecutive_up'] = df['up_bar'].groupby((df['up_bar'] != df['up_bar'].shift()).cumsum()).cumsum()
    df['consecutive_down'] = (1 - df['up_bar']).groupby(((1 - df['up_bar']) != (1 - df['up_bar']).shift()).cumsum()).cumsum()

    # Inside/outside bars
    df['inside_bar'] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    df['outside_bar'] = ((h > h.shift(1)) & (l < l.shift(1))).astype(int)

    return df


def prepare_ml_data(df, target_bars_ahead=10, target_type='classification',
                    min_move_pct=0.1):
    """
    Prepare features and targets for ML.

    target_type: 'classification' for buy/sell/hold, 'regression' for return prediction
    target_bars_ahead: how many bars to look ahead for the target
    min_move_pct: minimum move % to classify as buy/sell (rest is hold)
    """
    df = compute_features(df)

    # Create target
    future_return = df['Close'].shift(-target_bars_ahead) / df['Close'] - 1
    future_return_pct = future_return * 100

    if target_type == 'classification':
        # 1 = long, -1 = short, 0 = hold
        df['target'] = 0
        df.loc[future_return_pct > min_move_pct, 'target'] = 1
        df.loc[future_return_pct < -min_move_pct, 'target'] = -1
    else:
        df['target'] = future_return_pct

    # Drop NaN rows
    df = df.dropna()

    # Get feature columns (exclude OHLCV and target)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target',
                    'Adj Close', 'Dividends', 'Stock Splits', 'Capital Gains',
                    'Repaired?']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    return df, feature_cols
