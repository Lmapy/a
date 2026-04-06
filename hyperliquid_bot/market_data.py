"""Market data fetching from Hyperliquid and CoinGlass."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from hyperliquid.info import Info

from .config import BotConfig

logger = logging.getLogger(__name__)


@dataclass
class AssetInfo:
    symbol: str
    mark_price: float
    mid_price: float
    open_interest: float
    funding_rate: float
    volume_24h: float
    prev_day_price: float
    price_change_pct: float
    max_leverage: int
    sz_decimals: int


@dataclass
class OrderBookSnapshot:
    symbol: str
    bids: list[tuple[float, float]]  # (price, size)
    asks: list[tuple[float, float]]
    spread: float
    spread_pct: float
    bid_depth: float
    ask_depth: float
    imbalance: float  # positive = more bids, negative = more asks


@dataclass
class LiquidityData:
    symbol: str
    liquidation_levels: dict[str, float]
    long_short_ratio: float
    open_interest_change: float
    funding_rate_aggregated: float


class HyperliquidData:
    """Fetches and processes market data from Hyperliquid."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.info = Info(config.hyperliquid.base_url, skip_ws=True)
        self._asset_cache: dict[str, AssetInfo] = {}
        self._candle_cache: dict[str, pd.DataFrame] = {}
        self._last_refresh = 0.0

    def get_all_assets(self) -> list[AssetInfo]:
        """Fetch metadata and current state for all perp assets."""
        try:
            meta_and_ctxs = self.info.meta_and_asset_ctxs()
            meta = meta_and_ctxs[0]
            asset_ctxs = meta_and_ctxs[1]
            universe = meta["universe"]

            assets = []
            for i, coin_info in enumerate(universe):
                if i >= len(asset_ctxs):
                    break
                ctx = asset_ctxs[i]
                symbol = coin_info["name"]
                mark_px = float(ctx.get("markPx", 0))
                mid_px = float(ctx.get("midPx", 0)) if ctx.get("midPx") else mark_px
                prev_day_px = float(ctx.get("prevDayPx", 0))
                volume_24h = float(ctx.get("dayNtlVlm", 0))
                funding = float(ctx.get("funding", 0))
                oi = float(ctx.get("openInterest", 0))
                max_lev = int(coin_info.get("maxLeverage", 50))

                price_change = ((mark_px - prev_day_px) / prev_day_px * 100) if prev_day_px > 0 else 0

                asset = AssetInfo(
                    symbol=symbol,
                    mark_price=mark_px,
                    mid_price=mid_px,
                    open_interest=oi,
                    funding_rate=funding,
                    volume_24h=volume_24h,
                    prev_day_price=prev_day_px,
                    price_change_pct=price_change,
                    max_leverage=max_lev,
                    sz_decimals=coin_info.get("szDecimals", 2),
                )
                assets.append(asset)
                self._asset_cache[symbol] = asset

            self._last_refresh = time.time()
            logger.info(f"Fetched {len(assets)} assets from Hyperliquid")
            return assets

        except Exception as e:
            logger.error(f"Failed to fetch assets: {e}")
            return []

    def get_tradeable_assets(self) -> list[AssetInfo]:
        """Get assets filtered by minimum volume threshold."""
        assets = self.get_all_assets()
        min_vol = self.config.risk.min_volume_24h
        tradeable = [a for a in assets if a.volume_24h >= min_vol and a.mark_price > 0]
        logger.info(f"{len(tradeable)}/{len(assets)} assets meet volume threshold (${min_vol:,.0f})")
        return tradeable

    def get_orderbook(self, symbol: str) -> OrderBookSnapshot | None:
        """Get L2 orderbook snapshot for a symbol."""
        try:
            l2 = self.info.l2_snapshot(symbol)
            levels = l2.get("levels", [[], []])
            bids = [(float(b["px"]), float(b["sz"])) for b in levels[0][:20]]
            asks = [(float(a["px"]), float(a["sz"])) for a in levels[1][:20]]

            if not bids or not asks:
                return None

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            mid = (best_ask + best_bid) / 2
            spread_pct = (spread / mid * 100) if mid > 0 else float("inf")

            bid_depth = sum(px * sz for px, sz in bids)
            ask_depth = sum(px * sz for px, sz in asks)
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

            return OrderBookSnapshot(
                symbol=symbol,
                bids=bids,
                asks=asks,
                spread=spread,
                spread_pct=spread_pct,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                imbalance=imbalance,
            )
        except Exception as e:
            logger.warning(f"Failed to get orderbook for {symbol}: {e}")
            return None

    def get_candles(self, symbol: str, interval: str = "1h", lookback_hours: int = 168) -> pd.DataFrame:
        """Fetch OHLCV candle data for a symbol."""
        try:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - (lookback_hours * 3600 * 1000)
            raw = self.info.candles_snapshot(symbol, interval, start_ms, now_ms)

            if not raw:
                return pd.DataFrame()

            df = pd.DataFrame(raw)
            df["open"] = df["o"].astype(float)
            df["high"] = df["h"].astype(float)
            df["low"] = df["l"].astype(float)
            df["close"] = df["c"].astype(float)
            df["volume"] = df["v"].astype(float)
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
            df = df.reset_index(drop=True)

            self._candle_cache[symbol] = df
            return df

        except Exception as e:
            logger.warning(f"Failed to get candles for {symbol}: {e}")
            return pd.DataFrame()

    def get_funding_rates(self, symbol: str, hours: int = 72) -> list[dict]:
        """Fetch recent funding rate history."""
        try:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - (hours * 3600 * 1000)
            return self.info.funding_history(symbol, start_ms, now_ms)
        except Exception as e:
            logger.warning(f"Failed to get funding for {symbol}: {e}")
            return []

    def get_all_mids(self) -> dict[str, float]:
        """Get mid prices for all assets."""
        try:
            mids = self.info.all_mids()
            return {k: float(v) for k, v in mids.items()}
        except Exception as e:
            logger.error(f"Failed to get all mids: {e}")
            return {}

    def get_user_state(self, address: str) -> dict:
        """Get user account state including positions and margin."""
        try:
            return self.info.user_state(address)
        except Exception as e:
            logger.error(f"Failed to get user state: {e}")
            return {}


class CoinGlassData:
    """Fetches liquidity and sentiment data from CoinGlass API."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.session = requests.Session()
        if config.coinglass.enabled:
            self.session.headers.update({
                "coinglassSecret": config.coinglass.api_key,
                "Content-Type": "application/json",
            })

    @property
    def enabled(self) -> bool:
        return self.config.coinglass.enabled

    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        if not self.enabled:
            return None
        try:
            url = f"{self.config.coinglass.base_url}{endpoint}"
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "0" or data.get("success"):
                return data.get("data")
            logger.warning(f"CoinGlass API error: {data.get('msg', 'unknown')}")
            return None
        except Exception as e:
            logger.warning(f"CoinGlass request failed: {e}")
            return None

    def get_liquidation_data(self, symbol: str) -> dict | None:
        """Get aggregated liquidation data for a symbol."""
        return self._get("/futures/liquidation/aggregated", {"symbol": symbol, "timeType": 2})

    def get_long_short_ratio(self, symbol: str) -> float:
        """Get the global long/short ratio for a symbol."""
        data = self._get("/futures/globalLongShortAccountRatio/history", {
            "symbol": symbol,
            "timeType": "h4",
        })
        if data and isinstance(data, list) and len(data) > 0:
            latest = data[-1]
            return float(latest.get("longShortRatio", 1.0))
        return 1.0

    def get_open_interest_history(self, symbol: str) -> list[dict]:
        """Get open interest history."""
        data = self._get("/futures/openInterest/history", {
            "symbol": symbol,
            "timeType": "h4",
        })
        return data if isinstance(data, list) else []

    def get_funding_rates(self, symbol: str) -> dict | None:
        """Get aggregated funding rates across exchanges."""
        return self._get("/futures/funding/current", {"symbol": symbol})

    def get_liquidation_heatmap(self, symbol: str) -> dict | None:
        """Get liquidation heatmap data showing concentrated liquidation levels."""
        return self._get("/futures/liquidation/heatmap", {"symbol": symbol, "timeType": 2})

    def get_liquidity_score(self, symbol: str) -> dict:
        """Compute a composite liquidity score from CoinGlass data."""
        score = {"symbol": symbol, "total": 0.5, "components": {}}

        ls_ratio = self.get_long_short_ratio(symbol)
        # Extreme ratios suggest crowded trades = opportunity
        crowding_score = abs(ls_ratio - 1.0) / max(ls_ratio, 1.0)
        crowding_score = min(crowding_score, 1.0)
        score["components"]["crowding"] = crowding_score

        oi_data = self.get_open_interest_history(symbol)
        if len(oi_data) >= 2:
            recent_oi = float(oi_data[-1].get("openInterest", 0))
            prev_oi = float(oi_data[-2].get("openInterest", 0))
            oi_change = (recent_oi - prev_oi) / prev_oi if prev_oi > 0 else 0
            score["components"]["oi_momentum"] = min(abs(oi_change), 1.0)
        else:
            score["components"]["oi_momentum"] = 0.0

        score["components"]["long_short_ratio"] = ls_ratio

        # Weighted composite
        score["total"] = (
            crowding_score * 0.5
            + score["components"]["oi_momentum"] * 0.5
        )
        return score
