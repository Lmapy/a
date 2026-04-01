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
    Fill,
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
from src.strategy.level_sweep import LevelSweepStrategy
from src.strategy.supply_demand import SupplyDemandStrategy

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
        broker = PaperBroker(slippage_ticks=0.0, commission_per_contract=2.50)
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

            # Reset strategy daily state
            for strat in self.strategies.values():
                if hasattr(strat, 'reset_daily'):
                    strat.reset_daily()

            day_pnl = 0.0
            day_trades: list[TradeRecord] = []

            pending_entry_signal = None  # Holds signal to fill on NEXT bar

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

                # 0. Fill pending entry from previous bar's signal on THIS bar's open
                if pending_entry_signal is not None:
                    sig = pending_entry_signal
                    pending_entry_signal = None

                    if not risk_engine.lockout.is_locked() and bar.instrument not in risk_engine.positions:
                        # Adjust stop/tp relative to actual fill price (bar.open)
                        price_diff = bar.open - sig.entry_price
                        adjusted_stop = sig.stop_loss + price_diff
                        adjusted_tp = sig.take_profit + price_diff if sig.take_profit else None

                        # Use adjusted stop for sizing (distance from fill price)
                        stop_ticks = calculate_stop_ticks(
                            bar.open, adjusted_stop, instrument
                        )

                        if stop_ticks > 0:
                            size = risk_engine.get_position_size(instrument, stop_ticks)
                            alloc_pct = sig.confidence  # reuse confidence for alloc
                            size = max(1, int(size * alloc_pct)) if size > 0 else 0

                            if size > 0:
                                order = create_order_from_signal(sig, size)
                                approval = risk_engine.approve_order(order)
                                if approval.approved:
                                    order.quantity = approval.quantity
                                    broker.submit_order(order)
                                    fills = broker.process_bar(bar)
                                    for fill in fills:
                                        risk_engine.on_fill(fill, sig.strategy_name)
                                        if bar.instrument in risk_engine.positions:
                                            pos = risk_engine.positions[bar.instrument]
                                            pos.stop_loss = adjusted_stop
                                            pos.take_profit = adjusted_tp

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
                                # Use the actual stop/TP price as exit, not bar.open
                                # Add 1 tick slippage against us on stop exits
                                exit_reason = exit_signal.metadata.get("exit_reason", "")
                                spec = INSTRUMENT_SPECS[instrument]
                                exit_price = exit_signal.entry_price
                                if exit_reason in ("stop_loss", "trailing_stop"):
                                    # Stops slip 1 tick against us
                                    if pos.direction == SignalDirection.LONG:
                                        exit_price -= spec["tick_size"]
                                    else:
                                        exit_price += spec["tick_size"]

                                # Create a synthetic fill at the correct price
                                commission = 2.50 * pos.quantity
                                fake_fill = Fill(
                                    order_id="exit",
                                    instrument=instrument,
                                    direction=(SignalDirection.SHORT if pos.direction == SignalDirection.LONG
                                               else SignalDirection.LONG),
                                    quantity=pos.quantity,
                                    fill_price=exit_price,
                                    timestamp=bar.timestamp,
                                    commission=commission,
                                    slippage=0,
                                )
                                pnl = risk_engine.on_fill(fake_fill, strat_name)
                                if pnl is not None:
                                    day_pnl += pnl
                                    day_trades.append(TradeRecord(
                                        instrument=instrument,
                                        direction=pos.direction,
                                        quantity=pos.quantity,
                                        entry_price=pos.entry_price,
                                        exit_price=exit_price,
                                        entry_time=pos.entry_time,
                                        exit_time=bar.timestamp,
                                        pnl=pnl,
                                        commission=commission,
                                        strategy_name=strat_name,
                                    ))

                # 4. Generate entry signals (will be filled on NEXT bar)
                if not risk_engine.lockout.is_locked() and bar.instrument not in risk_engine.positions:
                    if pending_entry_signal is None:
                        for strat_name, (strat, alloc_pct) in active.items():
                            signal = strat.on_bar(bar, indicators, regime)
                            if signal:
                                # Store allocation pct in confidence for later use
                                signal.confidence = alloc_pct
                                pending_entry_signal = signal
                                break  # only one pending signal at a time

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
