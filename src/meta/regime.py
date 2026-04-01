from __future__ import annotations

import pandas as pd

from src.data.models import MarketRegime


class RegimeDetector:
    """Classifies current market regime using ADX, Bollinger Bandwidth, and EMA slope.

    Algorithm:
        ADX > 35 + clear EMA direction → STRONG_TREND_UP/DOWN
        ADX 20-35                     → WEAK_TREND
        ADX < 20 + narrow BBands      → LOW_VOLATILITY / RANGING
        ADX < 20 + wide BBands        → HIGH_VOLATILITY (choppy)
    """

    def __init__(self, adx_strong: float = 35.0, adx_weak: float = 20.0,
                 bbw_high_pct: float = 75.0, bbw_low_pct: float = 25.0,
                 lookback: int = 100):
        self.adx_strong = adx_strong
        self.adx_weak = adx_weak
        self.bbw_high_pct = bbw_high_pct
        self.bbw_low_pct = bbw_low_pct
        self.lookback = lookback
        self._bbw_history: list[float] = []

    def detect(self, indicators: pd.Series) -> MarketRegime:
        adx_val = indicators.get("adx_14")
        plus_di = indicators.get("plus_di")
        minus_di = indicators.get("minus_di")
        bb_width = indicators.get("bb_width")
        ema_20 = indicators.get("ema_20")
        ema_50 = indicators.get("ema_50")

        if any(v is None or (isinstance(v, float) and pd.isna(v))
               for v in [adx_val, bb_width, ema_20, ema_50]):
            return MarketRegime.RANGING  # default

        # Track BB width history for percentile calculation
        self._bbw_history.append(bb_width)
        if len(self._bbw_history) > self.lookback:
            self._bbw_history = self._bbw_history[-self.lookback:]

        # BB width percentile
        bbw_percentile = self._bbw_percentile(bb_width)

        # Strong trend
        if adx_val >= self.adx_strong:
            if ema_20 > ema_50:
                return MarketRegime.STRONG_TREND_UP
            else:
                return MarketRegime.STRONG_TREND_DOWN

        # Weak trend
        if adx_val >= self.adx_weak:
            if bbw_percentile >= self.bbw_high_pct:
                return MarketRegime.HIGH_VOLATILITY
            return MarketRegime.WEAK_TREND

        # Low ADX - ranging or volatile
        if bbw_percentile >= self.bbw_high_pct:
            return MarketRegime.HIGH_VOLATILITY
        if bbw_percentile <= self.bbw_low_pct:
            return MarketRegime.LOW_VOLATILITY

        return MarketRegime.RANGING

    def _bbw_percentile(self, current_bbw: float) -> float:
        if len(self._bbw_history) < 10:
            return 50.0
        count_below = sum(1 for x in self._bbw_history if x <= current_bbw)
        return (count_below / len(self._bbw_history)) * 100.0
