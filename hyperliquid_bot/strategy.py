"""Trading strategy engine with momentum, mean-reversion, and liquidity signals."""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .market_data import AssetInfo, CoinGlassData, HyperliquidData, OrderBookSnapshot

logger = logging.getLogger(__name__)


class Signal(Enum):
    STRONG_LONG = 2
    LONG = 1
    NEUTRAL = 0
    SHORT = -1
    STRONG_SHORT = -2


@dataclass
class AssetSignal:
    symbol: str
    signal: Signal
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    components: dict = field(default_factory=dict)
    suggested_leverage: int = 1
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


def compute_technical_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe."""
    if df.empty or len(df) < config.ema_slow + 5:
        return df

    # EMAs
    df["ema_fast"] = df["close"].ewm(span=config.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=config.ema_slow, adjust=False).mean()
    df["ema_trend"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / config.rsi_period, min_periods=config.rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / config.rsi_period, min_periods=config.rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(config.bb_period).mean()
    bb_std = df["close"].rolling(config.bb_period).std()
    df["bb_upper"] = df["bb_mid"] + config.bb_std * bb_std
    df["bb_lower"] = df["bb_mid"] - config.bb_std * bb_std
    bb_width = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_width.replace(0, np.nan)

    # ATR for volatility
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(config.atr_period).mean()
    df["atr_pct"] = df["atr"] / df["close"] * 100

    # Volume analysis
    df["volume_ma"] = df["volume"].rolling(config.volume_ma_period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Rate of change
    df["roc_5"] = df["close"].pct_change(5) * 100
    df["roc_20"] = df["close"].pct_change(20) * 100

    return df


class StrategyEngine:
    """Multi-signal strategy combining momentum, mean-reversion, and liquidity."""

    def __init__(self, config: StrategyConfig, market_data: HyperliquidData, coinglass: CoinGlassData):
        self.config = config
        self.market_data = market_data
        self.coinglass = coinglass

    def _momentum_score(self, df: pd.DataFrame, asset: AssetInfo) -> tuple[float, dict]:
        """Score based on trend-following signals."""
        if df.empty or "ema_trend" not in df.columns:
            return 0.0, {}

        last = df.iloc[-1]
        components = {}

        # EMA crossover direction
        ema_score = float(np.clip(last["ema_trend"] * 20, -1, 1))
        components["ema_trend"] = ema_score

        # MACD histogram direction and strength
        if "macd_hist" in df.columns and not np.isnan(last["macd_hist"]):
            macd_norm = float(np.clip(last["macd_hist"] / (last["close"] * 0.01), -1, 1))
            components["macd"] = macd_norm
        else:
            macd_norm = 0.0

        # ROC momentum
        if "roc_5" in df.columns and not np.isnan(last["roc_5"]):
            roc_score = float(np.clip(last["roc_5"] / 5, -1, 1))
            components["roc"] = roc_score
        else:
            roc_score = 0.0

        # Volume confirmation: strong moves with volume are more trustworthy
        vol_confirm = 1.0
        if "volume_ratio" in df.columns and not np.isnan(last["volume_ratio"]):
            vol_confirm = min(last["volume_ratio"], 3.0) / 3.0
            components["volume_confirm"] = vol_confirm

        score = (ema_score * 0.4 + macd_norm * 0.35 + roc_score * 0.25) * (0.5 + 0.5 * vol_confirm)
        return float(np.clip(score, -1, 1)), components

    def _mean_reversion_score(self, df: pd.DataFrame, asset: AssetInfo) -> tuple[float, dict]:
        """Score based on mean-reversion / oversold-overbought signals."""
        if df.empty or "rsi" not in df.columns:
            return 0.0, {}

        last = df.iloc[-1]
        components = {}

        # RSI-based score: oversold = long opportunity, overbought = short opportunity
        rsi = last["rsi"]
        if not np.isnan(rsi):
            if rsi < self.config.rsi_oversold:
                rsi_score = (self.config.rsi_oversold - rsi) / self.config.rsi_oversold
            elif rsi > self.config.rsi_overbought:
                rsi_score = -(rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
            else:
                rsi_score = 0.0
            components["rsi"] = float(rsi_score)
        else:
            rsi_score = 0.0

        # Bollinger Band position: near lower band = long, near upper = short
        if "bb_position" in df.columns and not np.isnan(last["bb_position"]):
            bb_pos = last["bb_position"]
            bb_score = -(bb_pos - 0.5) * 2  # Maps [0,1] -> [1,-1]
            bb_score = float(np.clip(bb_score, -1, 1))
            components["bb_position"] = bb_score
        else:
            bb_score = 0.0

        # Funding rate mean reversion: high positive funding = potential short
        funding = asset.funding_rate
        funding_score = float(np.clip(-funding * 100, -1, 1))
        components["funding_reversion"] = funding_score

        score = rsi_score * 0.45 + bb_score * 0.35 + funding_score * 0.20
        return float(np.clip(score, -1, 1)), components

    def _liquidity_score(self, asset: AssetInfo, orderbook: OrderBookSnapshot | None) -> tuple[float, dict]:
        """Score based on orderbook liquidity and CoinGlass data."""
        components = {}
        scores = []

        # Orderbook imbalance: strong bid side = bullish
        if orderbook:
            imb_score = float(np.clip(orderbook.imbalance * 2, -1, 1))
            components["ob_imbalance"] = imb_score
            scores.append(imb_score * 0.4)

            # Tight spreads are better for trading
            spread_quality = max(0, 1 - orderbook.spread_pct / 0.5)
            components["spread_quality"] = spread_quality
        else:
            scores.append(0.0)

        # CoinGlass liquidity data
        if self.coinglass.enabled:
            cg_score = self.coinglass.get_liquidity_score(asset.symbol)
            ls_ratio = cg_score["components"].get("long_short_ratio", 1.0)

            # Contrarian: if too many longs, favor short and vice versa
            if ls_ratio > 1.5:
                crowd_signal = -(ls_ratio - 1) / ls_ratio
            elif ls_ratio < 0.67:
                crowd_signal = (1 - ls_ratio) / 1
            else:
                crowd_signal = 0.0
            crowd_signal = float(np.clip(crowd_signal, -1, 1))
            components["crowd_contrarian"] = crowd_signal
            scores.append(crowd_signal * 0.3)

            oi_mom = cg_score["components"].get("oi_momentum", 0)
            components["oi_momentum"] = oi_mom
            scores.append(oi_mom * 0.3)
        else:
            scores.append(0.0)
            scores.append(0.0)

        total = sum(scores)
        return float(np.clip(total, -1, 1)), components

    def _funding_score(self, asset: AssetInfo) -> tuple[float, dict]:
        """Score based on funding rate arbitrage opportunity."""
        funding = asset.funding_rate
        components = {}

        # High positive funding = pay to be long, earn to be short
        # High negative funding = earn to be long, pay to be short
        # We lean toward the earning side
        score = float(np.clip(-funding * 50, -1, 1))
        components["funding_rate"] = asset.funding_rate
        components["funding_signal"] = score

        return score, components

    def analyze_asset(self, asset: AssetInfo) -> AssetSignal:
        """Generate a composite trading signal for an asset."""
        df = self.market_data.get_candles(asset.symbol, "1h", 168)
        df = compute_technical_indicators(df, self.config)
        orderbook = self.market_data.get_orderbook(asset.symbol)

        mom_score, mom_components = self._momentum_score(df, asset)
        mr_score, mr_components = self._mean_reversion_score(df, asset)
        liq_score, liq_components = self._liquidity_score(asset, orderbook)
        fund_score, fund_components = self._funding_score(asset)

        # Weighted composite based on strategy mode
        if self.config.mode == "momentum":
            weights = {"momentum": 0.6, "mean_reversion": 0.15, "liquidity": 0.15, "funding": 0.1}
        elif self.config.mode == "mean_reversion":
            weights = {"momentum": 0.15, "mean_reversion": 0.6, "liquidity": 0.15, "funding": 0.1}
        elif self.config.mode == "liquidity":
            weights = {"momentum": 0.15, "mean_reversion": 0.15, "liquidity": 0.5, "funding": 0.2}
        else:  # hybrid
            weights = {
                "momentum": self.config.momentum_weight,
                "mean_reversion": self.config.mean_reversion_weight,
                "liquidity": self.config.liquidity_weight,
                "funding": self.config.funding_weight,
            }

        composite = (
            mom_score * weights["momentum"]
            + mr_score * weights["mean_reversion"]
            + liq_score * weights["liquidity"]
            + fund_score * weights["funding"]
        )

        # Confidence based on signal agreement
        signals = [mom_score, mr_score, liq_score, fund_score]
        signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in signals]
        agreement = abs(sum(signs)) / len(signs)
        avg_magnitude = np.mean([abs(s) for s in signals])
        confidence = float(agreement * 0.6 + avg_magnitude * 0.4)

        # Determine signal direction
        if composite > 0.5:
            signal = Signal.STRONG_LONG
        elif composite > self.config.min_score_threshold:
            signal = Signal.LONG
        elif composite < -0.5:
            signal = Signal.STRONG_SHORT
        elif composite < -self.config.min_score_threshold:
            signal = Signal.SHORT
        else:
            signal = Signal.NEUTRAL

        # Compute stop loss and take profit from ATR
        atr_pct = 2.0
        if not df.empty and "atr_pct" in df.columns:
            last_atr = df.iloc[-1]["atr_pct"]
            if not np.isnan(last_atr):
                atr_pct = last_atr

        if signal in (Signal.LONG, Signal.STRONG_LONG):
            stop_loss = asset.mark_price * (1 - atr_pct * 1.5 / 100)
            take_profit = asset.mark_price * (1 + atr_pct * 3 / 100)
        elif signal in (Signal.SHORT, Signal.STRONG_SHORT):
            stop_loss = asset.mark_price * (1 + atr_pct * 1.5 / 100)
            take_profit = asset.mark_price * (1 - atr_pct * 3 / 100)
        else:
            stop_loss = 0.0
            take_profit = 0.0

        # Suggested leverage based on volatility (lower vol = can use more leverage safely)
        if atr_pct < 1.5:
            suggested_lev = 3
        elif atr_pct < 3.0:
            suggested_lev = 2
        else:
            suggested_lev = 1

        return AssetSignal(
            symbol=asset.symbol,
            signal=signal,
            score=float(np.clip(composite, -1, 1)),
            confidence=confidence,
            components={
                "momentum": {"score": mom_score, **mom_components},
                "mean_reversion": {"score": mr_score, **mr_components},
                "liquidity": {"score": liq_score, **liq_components},
                "funding": {"score": fund_score, **fund_components},
            },
            suggested_leverage=suggested_lev,
            entry_price=asset.mark_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def rank_assets(self, assets: list[AssetInfo]) -> list[AssetSignal]:
        """Analyze and rank all assets by signal strength."""
        signals = []
        for asset in assets:
            try:
                sig = self.analyze_asset(asset)
                if sig.signal != Signal.NEUTRAL:
                    signals.append(sig)
            except Exception as e:
                logger.warning(f"Failed to analyze {asset.symbol}: {e}")

        # Sort by absolute score * confidence
        signals.sort(key=lambda s: abs(s.score) * s.confidence, reverse=True)
        logger.info(f"Ranked {len(signals)} actionable signals from {len(assets)} assets")
        return signals
