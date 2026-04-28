"""ATR trend-following strategy with volatility-adjusted position sizing.

Entry: EMA(fast) crosses EMA(slow) in the trend direction.
Size:  risk_dollars / (ATR * tick_value_per_point) contracts, capped by tier max.
Stop:  hard stop at stop_atr * ATR from entry; trail stop after 2×ATR profit.
Exit:  stop hit, or opposite EMA cross, or daily session end.

This is the closest thing to a systematic trend-follower used by funded traders.
"""
from __future__ import annotations

from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.engine.broker import Signal
from prop_backtest.firms.base import AccountTier
from prop_backtest.strategy.base import Strategy


def _ema(values: list[float], period: int) -> float:
    """Exponential moving average of the last `period` values (fast calc)."""
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _atr(history: list[BarData], period: int) -> float:
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


class ATRTrend(Strategy):
    """ATR-sized EMA trend-following strategy.

    Parameters:
        fast:         Fast EMA period (default 9).
        slow:         Slow EMA period (default 21).
        atr_period:   ATR lookback (default 14).
        risk_dollars: Dollar risk per trade (default $200). Used to size contracts.
        stop_atr:     Initial stop distance in ATR multiples (default 1.5).
        trail_atr:    Trail stop kicks in after this many ATRs of profit (default 2.0).
        contracts:    Fixed contracts override; set to None for dynamic sizing.
        risk_aware:   Halve size when near drawdown floor (default True).
    """

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        atr_period: int = 14,
        risk_dollars: float = 200.0,
        stop_atr: float = 1.5,
        trail_atr: float = 2.0,
        contracts: int | None = None,
        risk_aware: bool = True,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow})")
        self.fast         = fast
        self.slow         = slow
        self.atr_period   = atr_period
        self.risk_dollars = risk_dollars
        self.stop_atr     = stop_atr
        self.trail_atr    = trail_atr
        self.contracts    = contracts
        self.risk_aware   = risk_aware

        self._stop_price:  float | None = None
        self._entry_price: float | None = None
        self._entry_atr:   float | None = None
        self._trailing:    bool = False

    def on_start(self, contract: ContractSpec, tier: AccountTier) -> None:
        self._contract = contract
        self._tier     = tier

    def on_bar(
        self,
        bar: BarData,
        history: list[BarData],
        account: AccountState,
    ) -> Signal:
        min_bars = max(self.slow, self.atr_period) + 2
        if len(history) < min_bars:
            return Signal(action="hold")

        closes   = [b.close for b in history]
        ema_fast = _ema(closes[-self.fast * 3:], self.fast)
        ema_slow = _ema(closes[-self.slow * 3:], self.slow)
        atr      = _atr(history, self.atr_period)
        pos      = account.position_contracts

        # ── Manage open position ───────────────────────────────────────────
        if pos != 0 and self._stop_price is not None:
            # Activate trailing stop once profit >= trail_atr * ATR
            if not self._trailing and self._entry_price and self._entry_atr:
                profit_ticks = (
                    (bar.close - self._entry_price) if pos > 0
                    else (self._entry_price - bar.close)
                )
                if profit_ticks >= self.trail_atr * self._entry_atr:
                    self._trailing = True

            # Update trail stop
            if self._trailing and self._entry_atr:
                if pos > 0:
                    new_stop = bar.close - self._entry_atr
                    if new_stop > self._stop_price:
                        self._stop_price = new_stop
                else:
                    new_stop = bar.close + self._entry_atr
                    if new_stop < self._stop_price:
                        self._stop_price = new_stop

            # Stop hit
            if pos > 0 and bar.close <= self._stop_price:
                self._reset_trade()
                return Signal(action="close")
            if pos < 0 and bar.close >= self._stop_price:
                self._reset_trade()
                return Signal(action="close")

            # EMA reversal exit
            if pos > 0 and ema_fast < ema_slow:
                self._reset_trade()
                return Signal(action="close")
            if pos < 0 and ema_fast > ema_slow:
                self._reset_trade()
                return Signal(action="close")

        # ── Entry ──────────────────────────────────────────────────────────
        if pos == 0:
            size = self._get_contracts(account, atr)
            if size < 1:
                return Signal(action="hold")

            if ema_fast > ema_slow:
                self._entry_price = bar.close
                self._entry_atr   = atr
                self._stop_price  = bar.close - self.stop_atr * atr
                self._trailing    = False
                return Signal(action="buy", contracts=size)

            if ema_fast < ema_slow:
                self._entry_price = bar.close
                self._entry_atr   = atr
                self._stop_price  = bar.close + self.stop_atr * atr
                self._trailing    = False
                return Signal(action="sell", contracts=size)

        return Signal(action="hold")

    def _reset_trade(self) -> None:
        self._stop_price  = None
        self._entry_price = None
        self._entry_atr   = None
        self._trailing    = False

    def _get_contracts(self, account: AccountState, atr: float) -> int:
        # Dynamic sizing: risk_dollars / dollar_value_of_one_ATR
        if self.contracts is not None:
            size = self.contracts
        else:
            atr_dollar_value = atr * self._contract.point_value
            size = max(1, int(self.risk_dollars / atr_dollar_value)) if atr_dollar_value > 0 else 1
            size = min(size, self._tier.max_contracts)

        if not self.risk_aware:
            return size
        p = account.dd_floor_proximity
        if p >= 0.5:
            return size
        elif p >= 0.25:
            return max(1, size // 2)
        return 1
