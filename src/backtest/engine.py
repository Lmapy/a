from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from src.backtest.metrics import compute_metrics
from src.core.config import FirmConfig, GlobalConfig, RiskConfig
from src.data.indicators import compute_indicators
from src.data.models import (
    Bar,
    ChallengeStatus,
    INSTRUMENT_SPECS,
    SignalDirection,
    TradeRecord,
)
from src.execution.order import calculate_stop_ticks, create_order_from_signal
from src.execution.paper import PaperBroker
from src.meta.allocator import RiskAllocator
from src.meta.regime import RegimeDetector
from src.meta.selector import StrategySelector
from src.risk.engine import RiskEngine
from src.strategy.base import BaseStrategy
from src.strategy.breakout import BreakoutStrategy

logger = logging.getLogger(__name__)


class BacktestResult:
    def __init__(self):
        self.trades: list[TradeRecord] = []
        self.daily_pnls: list[float] = []
        self.daily_balances: list[float] = []
        self.status: ChallengeStatus = ChallengeStatus.ACTIVE
        self.days_traded: int = 0
        self.final_balance: float = 0.0
        self.metrics: dict = {}


class BacktestEngine:
    """Event-driven backtesting engine that simulates a prop firm challenge."""

    def __init__(self, config: GlobalConfig, strategies: dict[str, BaseStrategy]):
        self.config = config
        self.strategies = strategies

    def run(self, data: pd.DataFrame, instrument: str = "ES") -> BacktestResult:
        """Run a backtest on OHLCV data.

        Args:
            data: DataFrame with columns: open, high, low, close, volume
                  and a DatetimeIndex.
            instrument: Futures instrument symbol.

        Returns:
            BacktestResult with trades, daily P&L, and challenge outcome.
        """
        result = BacktestResult()

        # Initialize components
        risk_engine = RiskEngine(self.config.firm, self.config.risk)
        broker = PaperBroker(slippage_ticks=1.0, commission_per_contract=2.50)
        regime_detector = RegimeDetector()
        selector = StrategySelector(self.strategies, cooldown_bars=self.config.meta.strategy_cooldown_bars)
        allocator = RiskAllocator()

        # Compute indicators
        session_starts = self._detect_session_starts(data)
        df = compute_indicators(data, session_starts)

        # Group bars by trading day
        daily_groups = self._group_by_day(df)

        for day_date, day_bars in daily_groups:
            if risk_engine.status != ChallengeStatus.ACTIVE:
                break

            risk_engine.start_session()

            # Reset breakout strategy daily state
            for strat in self.strategies.values():
                if isinstance(strat, BreakoutStrategy):
                    strat.reset_daily()

            day_pnl = 0.0
            day_trades: list[TradeRecord] = []

            for idx, row in day_bars.iterrows():
                bar = Bar(
                    timestamp=idx,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row["volume"]),
                    instrument=instrument,
                )

                indicators = row

                # 1. Detect regime
                regime = regime_detector.detect(indicators)

                # 2. Select strategies for this regime
                active = selector.select(regime)

                # 3. Check exits on open positions
                for strat_name, (strat, alloc_pct) in active.items():
                    if bar.instrument in risk_engine.positions:
                        pos = risk_engine.positions[bar.instrument]
                        if pos.strategy_name == strat_name:
                            exit_signal = strat.should_exit(pos, bar, indicators)
                            if exit_signal:
                                exit_order = create_order_from_signal(exit_signal, pos.quantity)
                                broker.submit_order(exit_order)
                                fills = broker.process_bar(bar)
                                for fill in fills:
                                    pnl = risk_engine.on_fill(fill, strat_name)
                                    if pnl is not None:
                                        day_pnl += pnl
                                        day_trades.append(TradeRecord(
                                            instrument=instrument,
                                            direction=pos.direction,
                                            quantity=pos.quantity,
                                            entry_price=pos.entry_price,
                                            exit_price=fill.fill_price,
                                            entry_time=pos.entry_time,
                                            exit_time=bar.timestamp,
                                            pnl=pnl,
                                            commission=fill.commission,
                                            strategy_name=strat_name,
                                        ))

                # 4. Check entries (skip if locked out or already in position)
                if not risk_engine.lockout.is_locked() and bar.instrument not in risk_engine.positions:
                    for strat_name, (strat, alloc_pct) in active.items():
                        if risk_engine.lockout.is_locked():
                            break
                        if bar.instrument in risk_engine.positions:
                            break

                        signal = strat.on_bar(bar, indicators, regime)
                        if signal:
                            stop_ticks = calculate_stop_ticks(
                                signal.entry_price, signal.stop_loss, instrument
                            )
                            if stop_ticks <= 0:
                                continue

                            size = risk_engine.get_position_size(instrument, stop_ticks)
                            # Apply allocation scaling
                            size = max(1, int(size * alloc_pct)) if size > 0 else 0

                            if size > 0:
                                order = create_order_from_signal(signal, size)
                                approval = risk_engine.approve_order(order)
                                if approval.approved:
                                    order.quantity = approval.quantity
                                    broker.submit_order(order)
                                    fills = broker.process_bar(bar)
                                    for fill in fills:
                                        risk_engine.on_fill(fill, strat_name)
                                        # Set stop/tp on position
                                        if bar.instrument in risk_engine.positions:
                                            pos = risk_engine.positions[bar.instrument]
                                            pos.stop_loss = signal.stop_loss
                                            pos.take_profit = signal.take_profit

                # 5. Update risk engine with current bar
                risk_engine.on_bar(bar)

            # End of day: close all positions
            last_row = day_bars.iloc[-1]
            close_prices = {instrument: last_row["close"]}

            # Flatten and record any remaining position P&L
            if instrument in risk_engine.positions:
                pos = risk_engine.positions[instrument]
                flatten_pnl = risk_engine.flatten_all(close_prices)
                day_pnl += flatten_pnl
                if flatten_pnl != 0:
                    day_trades.append(TradeRecord(
                        instrument=instrument,
                        direction=pos.direction,
                        quantity=pos.quantity,
                        entry_price=pos.entry_price,
                        exit_price=last_row["close"],
                        entry_time=pos.entry_time,
                        exit_time=day_bars.index[-1],
                        pnl=flatten_pnl,
                        commission=0,
                        strategy_name=pos.strategy_name,
                        metadata={"exit_reason": "eod_flatten"},
                    ))

            status = risk_engine.on_session_close()
            result.daily_pnls.append(day_pnl)
            result.daily_balances.append(risk_engine.tracker.current_balance)
            result.trades.extend(day_trades)
            result.days_traded += 1

            if status != ChallengeStatus.ACTIVE:
                result.status = status
                break

        if result.status == ChallengeStatus.ACTIVE:
            result.status = ChallengeStatus.ACTIVE  # didn't finish

        result.final_balance = risk_engine.tracker.current_balance
        result.metrics = compute_metrics(result.trades, result.daily_pnls)
        return result

    @staticmethod
    def _detect_session_starts(df: pd.DataFrame) -> pd.Series:
        """Detect session boundaries by checking when the date changes."""
        dates = df.index.date
        session_starts = pd.Series(False, index=df.index)
        if len(dates) > 0:
            session_starts.iloc[0] = True
            for i in range(1, len(dates)):
                if dates[i] != dates[i - 1]:
                    session_starts.iloc[i] = True
        return session_starts

    @staticmethod
    def _group_by_day(df: pd.DataFrame) -> list[tuple]:
        """Group DataFrame rows by trading day."""
        df = df.copy()
        df["_date"] = df.index.date
        groups = []
        for date, group in df.groupby("_date"):
            groups.append((date, group.drop(columns=["_date"])))
        return groups
