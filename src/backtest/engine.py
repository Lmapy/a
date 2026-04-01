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
    """Event-driven backtesting engine that simulates a prop firm challenge.

    Supports multiple instruments trading concurrently with independent
    positions per instrument, sharing a single risk budget.
    """

    def __init__(self, config: GlobalConfig, strategies: dict[str, BaseStrategy]):
        self.config = config
        self.strategies = strategies

    def run(self, data: pd.DataFrame, instrument: str = "ES") -> BacktestResult:
        """Run a single-instrument backtest. Convenience wrapper around run_multi."""
        return self.run_multi({instrument: data})

    def run_multi(self, instrument_data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Run a multi-instrument backtest on OHLCV data.

        Args:
            instrument_data: Dict of instrument -> DataFrame with OHLCV + DatetimeIndex.
                             Each instrument can have different trading hours.

        Returns:
            BacktestResult with trades, daily P&L, and challenge outcome.
        """
        result = BacktestResult()

        # Initialize components
        risk_engine = RiskEngine(self.config.firm, self.config.risk)
        broker = PaperBroker(slippage_ticks=0.0, commission_per_contract=2.50)
        regime_detector = RegimeDetector()
        selector = StrategySelector(self.strategies, cooldown_bars=self.config.meta.strategy_cooldown_bars)

        # Compute indicators for each instrument
        indicator_dfs: dict[str, pd.DataFrame] = {}
        for inst, df in instrument_data.items():
            session_starts = self._detect_session_starts(df)
            indicator_dfs[inst] = compute_indicators(df, session_starts)

        # Merge all instruments into a single timeline
        # Each bar gets tagged with its instrument
        all_bars = []
        for inst, idf in indicator_dfs.items():
            for idx, row in idf.iterrows():
                all_bars.append((idx, inst, row))

        # Sort by timestamp
        all_bars.sort(key=lambda x: x[0])

        # Group by trading day (based on calendar date)
        daily_groups = self._group_bars_by_day(all_bars)

        # Per-instrument pending signals (filled on next bar for that instrument)
        pending_signals: dict[str, object] = {}  # inst -> Signal

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
            pending_signals.clear()

            for timestamp, inst, row in day_bars:
                bar = Bar(
                    timestamp=timestamp,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row["volume"]),
                    instrument=inst,
                )

                indicators = row
                spec = INSTRUMENT_SPECS.get(inst, INSTRUMENT_SPECS["ES"])

                # 0. Fill pending entry for THIS instrument
                if inst in pending_signals:
                    sig = pending_signals.pop(inst)

                    if not risk_engine.lockout.is_locked() and inst not in risk_engine.positions:
                        price_diff = bar.open - sig.entry_price
                        adjusted_stop = sig.stop_loss + price_diff
                        adjusted_tp = sig.take_profit + price_diff if sig.take_profit else None

                        stop_ticks = calculate_stop_ticks(bar.open, adjusted_stop, inst)

                        if stop_ticks > 0:
                            size = risk_engine.get_position_size(inst, stop_ticks)
                            alloc_pct = sig.confidence
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
                                        if inst in risk_engine.positions:
                                            pos = risk_engine.positions[inst]
                                            pos.stop_loss = adjusted_stop
                                            pos.take_profit = adjusted_tp

                # 1. Detect regime (uses indicators from whichever instrument)
                regime = regime_detector.detect(indicators)

                # 2. Select strategies
                active = selector.select(regime)

                # 3. Check exits on position for this instrument
                if inst in risk_engine.positions:
                    pos = risk_engine.positions[inst]
                    for strat_name, (strat, alloc_pct) in active.items():
                        if pos.strategy_name == strat_name:
                            exit_signal = strat.should_exit(pos, bar, indicators)
                            if exit_signal:
                                exit_reason = exit_signal.metadata.get("exit_reason", "")
                                exit_price = exit_signal.entry_price
                                if exit_reason in ("stop_loss", "trailing_stop"):
                                    if pos.direction == SignalDirection.LONG:
                                        exit_price -= spec["tick_size"]
                                    else:
                                        exit_price += spec["tick_size"]

                                commission = 2.50 * pos.quantity
                                fake_fill = Fill(
                                    order_id="exit",
                                    instrument=inst,
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
                                        instrument=inst,
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
                            break  # only check the strategy that owns this position

                # 4. Generate entry signals for this instrument
                #    Allow entry even if another instrument has a position
                if not risk_engine.lockout.is_locked() and inst not in risk_engine.positions:
                    if inst not in pending_signals:
                        for strat_name, (strat, alloc_pct) in active.items():
                            signal = strat.on_bar(bar, indicators, regime)
                            if signal:
                                signal.confidence = alloc_pct
                                pending_signals[inst] = signal
                                break

                # 5. Update risk engine
                risk_engine.on_bar(bar)

            # End of day: flatten all positions across all instruments
            close_prices = {}
            for inst, idf in indicator_dfs.items():
                # Get last bar for this day for each instrument
                day_mask = idf.index.date == day_date
                if day_mask.any():
                    last_row = idf.loc[day_mask].iloc[-1]
                    close_prices[inst] = last_row["close"]

            # Record all position info before flattening
            open_positions = {
                inst: {
                    "direction": pos.direction,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time,
                    "strategy_name": pos.strategy_name,
                }
                for inst, pos in risk_engine.positions.items()
                if inst in close_prices
            }

            if open_positions:
                flatten_pnl = risk_engine.flatten_all(close_prices)
                day_pnl += flatten_pnl

                # Distribute P&L proportionally across positions
                # (flatten_all returns total, we need per-instrument)
                if len(open_positions) == 1:
                    inst = next(iter(open_positions))
                    info = open_positions[inst]
                    if flatten_pnl != 0:
                        day_trades.append(TradeRecord(
                            instrument=inst,
                            direction=info["direction"],
                            quantity=info["quantity"],
                            entry_price=info["entry_price"],
                            exit_price=close_prices[inst],
                            entry_time=info["entry_time"],
                            exit_time=datetime.combine(day_date, datetime.min.time()),
                            pnl=flatten_pnl,
                            commission=0,
                            strategy_name=info["strategy_name"],
                            metadata={"exit_reason": "eod_flatten"},
                        ))
                else:
                    # Multiple positions: calculate per-instrument P&L
                    for inst, info in open_positions.items():
                        spec = INSTRUMENT_SPECS.get(inst, INSTRUMENT_SPECS["ES"])
                        ticks = (close_prices[inst] - info["entry_price"]) / spec["tick_size"]
                        inst_pnl = ticks * spec["tick_value"] * info["quantity"] * info["direction"].value
                        if inst_pnl != 0:
                            day_trades.append(TradeRecord(
                                instrument=inst,
                                direction=info["direction"],
                                quantity=info["quantity"],
                                entry_price=info["entry_price"],
                                exit_price=close_prices[inst],
                                entry_time=info["entry_time"],
                                exit_time=datetime.combine(day_date, datetime.min.time()),
                                pnl=inst_pnl,
                                commission=0,
                                strategy_name=info["strategy_name"],
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
            result.status = ChallengeStatus.ACTIVE

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

    @staticmethod
    def _group_bars_by_day(all_bars: list) -> list[tuple]:
        """Group (timestamp, instrument, row) tuples by calendar date."""
        from itertools import groupby
        groups = []
        for day_date, bars_iter in groupby(all_bars, key=lambda x: x[0].date()):
            groups.append((day_date, list(bars_iter)))
        return groups
