"""Download real historical candle data from Hyperliquid's public API.

Run this first to populate data/ directory, then run the backtester.
No API key needed - Hyperliquid's /info endpoint is public.
"""

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_meta() -> dict:
    """Fetch exchange metadata (all available perp assets)."""
    resp = requests.post(HYPERLIQUID_INFO_URL, json={"type": "metaAndAssetCtxs"}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_candles(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch candle data for a single symbol in one chunk."""
    resp = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {"coin": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_candles_chunked(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch candles in 7-day chunks to avoid API limits."""
    chunk_ms = 7 * 24 * 3600 * 1000  # 7 days
    all_candles = []
    current = start_ms

    while current < end_ms:
        chunk_end = min(current + chunk_ms, end_ms)
        try:
            candles = fetch_candles(symbol, interval, current, chunk_end)
            all_candles.extend(candles)
            logger.debug(f"  {symbol}: fetched {len(candles)} candles from chunk")
        except Exception as e:
            logger.warning(f"  {symbol}: chunk fetch failed: {e}")

        current = chunk_end
        time.sleep(0.2)  # Rate limit courtesy

    # Deduplicate by timestamp
    seen = set()
    deduped = []
    for c in all_candles:
        t = c["t"]
        if t not in seen:
            seen.add(t)
            deduped.append(c)

    return sorted(deduped, key=lambda c: c["t"])


def candles_to_dataframe(raw: list[dict]) -> pd.DataFrame:
    """Convert raw candle JSON to a clean DataFrame with validation."""
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["open"] = pd.to_numeric(df["o"], errors="coerce")
    df["high"] = pd.to_numeric(df["h"], errors="coerce")
    df["low"] = pd.to_numeric(df["l"], errors="coerce")
    df["close"] = pd.to_numeric(df["c"], errors="coerce")
    df["volume"] = pd.to_numeric(df["v"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Data integrity checks ---
    n_before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < n_before:
        logger.warning(f"  Dropped {n_before - len(df)} rows with NaN prices")

    # Check for OHLC consistency: high >= open,close,low and low <= open,close,high
    bad_hl = df[(df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"]) |
                (df["low"] > df["open"]) | (df["low"] > df["close"])]
    if len(bad_hl) > 0:
        logger.warning(f"  {len(bad_hl)} candles with inconsistent OHLC (high < low, etc.) - clamping")
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    # Check for zero/negative prices
    bad_px = df[(df["close"] <= 0) | (df["open"] <= 0)]
    if len(bad_px) > 0:
        logger.warning(f"  Dropping {len(bad_px)} rows with zero/negative prices")
        df = df[(df["close"] > 0) & (df["open"] > 0)]

    # Check for duplicate timestamps
    dupes = df["timestamp"].duplicated()
    if dupes.any():
        logger.warning(f"  Dropping {dupes.sum()} duplicate timestamps")
        df = df[~dupes]

    # Check for gaps (missing candles)
    if len(df) >= 2:
        diffs = df["timestamp"].diff().dropna()
        median_diff = diffs.median()
        gaps = diffs[diffs > median_diff * 2]
        if len(gaps) > 0:
            logger.info(f"  {len(gaps)} gaps detected in candle data (>2x median interval)")

    return df.reset_index(drop=True)


def download_top_assets(
    n_assets: int = 20,
    interval: str = "1h",
    lookback_days: int = 90,
    min_volume_24h: float = 500_000,
) -> dict[str, Path]:
    """Download candle data for top N assets by volume.

    Returns dict mapping symbol -> CSV file path.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Hyperliquid metadata...")
    meta_ctxs = fetch_meta()
    universe = meta_ctxs[0]["universe"]
    ctxs = meta_ctxs[1]

    # Rank by 24h volume
    assets = []
    for i, coin in enumerate(universe):
        if i < len(ctxs):
            vol = float(ctxs[i].get("dayNtlVlm", 0))
            mark_px = float(ctxs[i].get("markPx", 0))
            assets.append({
                "symbol": coin["name"],
                "volume_24h": vol,
                "mark_price": mark_px,
                "sz_decimals": coin.get("szDecimals", 2),
                "max_leverage": coin.get("maxLeverage", 50),
            })

    assets.sort(key=lambda a: a["volume_24h"], reverse=True)
    eligible = [a for a in assets if a["volume_24h"] >= min_volume_24h and a["mark_price"] > 0]
    selected = eligible[:n_assets]

    logger.info(f"Selected {len(selected)} assets (from {len(eligible)} eligible, {len(assets)} total)")
    for a in selected:
        logger.info(f"  {a['symbol']:10s}  vol=${a['volume_24h']:>15,.0f}  px=${a['mark_price']:.4f}")

    # Save asset metadata
    meta_path = DATA_DIR / "asset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(selected, f, indent=2)
    logger.info(f"Saved asset metadata to {meta_path}")

    # Download candles for each
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (lookback_days * 24 * 3600 * 1000)
    saved = {}

    for i, asset in enumerate(selected):
        symbol = asset["symbol"]
        csv_path = DATA_DIR / f"{symbol}_{interval}_{lookback_days}d.csv"

        logger.info(f"[{i+1}/{len(selected)}] Downloading {symbol} ({interval}, {lookback_days}d)...")
        raw = fetch_candles_chunked(symbol, interval, start_ms, now_ms)
        df = candles_to_dataframe(raw)

        if df.empty:
            logger.warning(f"  {symbol}: no data returned, skipping")
            continue

        # Final validation summary
        hours_span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
        expected_candles = hours_span  # for 1h interval
        coverage = len(df) / expected_candles * 100 if expected_candles > 0 else 0

        logger.info(
            f"  {symbol}: {len(df)} candles, "
            f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}, "
            f"coverage={coverage:.1f}%"
        )

        if coverage < 80:
            logger.warning(f"  {symbol}: LOW COVERAGE ({coverage:.1f}%) - backtest results may be unreliable")

        df.to_csv(csv_path, index=False)
        saved[symbol] = csv_path

        time.sleep(0.3)

    logger.info(f"\nDownloaded {len(saved)} assets to {DATA_DIR}/")
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_top_assets(n_assets=20, interval="1h", lookback_days=90)
