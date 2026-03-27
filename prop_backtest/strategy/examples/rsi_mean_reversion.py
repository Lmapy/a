"""RSI mean-reversion strategy.

Buys when RSI drops below the oversold threshold (default 30) and sells
when RSI rises above the overbought threshold (default 70).
Exits when RSI reverts to the neutral zone (40–60).

ATR-based stop: exits immediately if price moves adversely by stop_atr * ATR.
"""
from __future__ import annotations

from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.engine.broker import Signal
from prop_backtest.firms.base import AccountTier
from prop_backtest.strategy.base import Strategy


def _rsi(closes: list[float], period: int) -> float:
    """Wilder RSI over the last `period` bars."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-period:])  / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(history: list[BarData], period: int) -> float:
    """Average True Range over the last `period` bars."""
    if len(history) < 2:
        return history[-1].high - history[-1].low if history else 1.0
    trs = []
    for i in range(1, len(history)):
        b, prev = history[i], history[i - 1]
        trs.append(max(b.high - b.low,
                       abs(b.high - prev.close),
                       abs(b.low  - prev.close)))
    window = trs[-period:]
    return sum(window) / len(window)


class RSIMeanReversion(Strategy):
    """RSI mean-reversion strategy with ATR stop.

    Parameters:
        rsi_period:   RSI lookback period (default 14).
        oversold:     RSI threshold to go long (default 30).
        overbought:   RSI threshold to go short (default 70).
        exit_neutral: RSI level at which to exit (default 50).
        atr_period:   ATR lookback for stop calculation (default 14).
        stop_atr:     Stop distance in ATR multiples (default 1.5).
        contracts:    Contracts per trade (default 1).
        risk_aware:   Scale down near drawdown floor (default True).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_neutral: float = 50.0,
        atr_period: int = 14,
        stop_atr: float = 1.5,
        contracts: int = 1,
        risk_aware: bool = True,
    ) -> None:
        self.rsi_period   = rsi_period
        self.oversold     = oversold
        self.overbought   = overbought
        self.exit_neutral = exit_neutral
        self.atr_period   = atr_period
        self.stop_atr     = stop_atr
        self.contracts    = contracts
        self.risk_aware   = risk_aware
        self._stop_price: float | None = None

    def on_start(self, contract: ContractSpec, tier: AccountTier) -> None:
        self._contract = contract
        self._tier     = tier

    def on_bar(
        self,
        bar: BarData,
        history: list[BarData],
        account: AccountState,
    ) -> Signal:
        min_bars = max(self.rsi_period, self.atr_period) + 2
        if len(history) < min_bars:
            return Signal(action="hold")

        closes = [b.close for b in history]
        rsi    = _rsi(closes, self.rsi_period)
        atr    = _atr(history, self.atr_period)
        pos    = account.position_contracts

        # ── Stop check (before entry logic) ───────────────────────────────
        if pos != 0 and self._stop_price is not None:
            if pos > 0 and bar.close <= self._stop_price:
                self._stop_price = None
                return Signal(action="close")
            if pos < 0 and bar.close >= self._stop_price:
                self._stop_price = None
                return Signal(action="close")

        # ── Exit on RSI neutral reversion ──────────────────────────────────
        if pos > 0 and rsi >= self.exit_neutral:
            self._stop_price = None
            return Signal(action="close")
        if pos < 0 and rsi <= self.exit_neutral:
            self._stop_price = None
            return Signal(action="close")

        # ── Entry ──────────────────────────────────────────────────────────
        if pos == 0:
            size = self._get_contracts(account)
            if rsi <= self.oversold:
                self._stop_price = bar.close - self.stop_atr * atr
                return Signal(action="buy", contracts=size)
            if rsi >= self.overbought:
                self._stop_price = bar.close + self.stop_atr * atr
                return Signal(action="sell", contracts=size)

        return Signal(action="hold")

    def _get_contracts(self, account: AccountState) -> int:
        if not self.risk_aware:
            return self.contracts
        p = account.dd_floor_proximity
        if p >= 0.5:
            return self.contracts
        elif p >= 0.25:
            return max(1, self.contracts // 2)
        return 1
