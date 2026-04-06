"""
Prop Firm Challenge Simulator.
Simulates realistic prop firm challenge rules:
- Account size: $100,000
- Profit target: 8% ($8,000) for Phase 1 / 5% for Phase 2
- Max daily drawdown: 5% ($5,000)
- Max total drawdown: 10% ($10,000)
- Minimum trading days: typically 5
- Must pass within time limit

This simulator runs bar-by-bar with realistic execution:
- Spread simulation (gold typically 2-4 pips)
- Commission per lot
- Slippage simulation
- No look-ahead bias
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TradeResult:
    entry_time: object
    exit_time: object
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    lot_size: float
    pnl_dollars: float
    pnl_pips: float
    duration_bars: int
    exit_reason: str  # 'tp', 'sl', 'signal', 'eod'


@dataclass
class DayResult:
    date: object
    starting_balance: float
    ending_balance: float
    pnl: float
    pnl_pct: float
    max_drawdown_intraday: float
    num_trades: int
    winning_trades: int
    losing_trades: int


@dataclass
class ChallengeResult:
    passed: bool
    phase: str
    reason: str  # 'target_hit', 'daily_dd_breach', 'total_dd_breach', 'time_expired'
    days_taken: int
    final_balance: float
    total_pnl: float
    total_pnl_pct: float
    max_daily_dd: float
    max_total_dd: float
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    daily_results: List[DayResult] = field(default_factory=list)
    trades: List[TradeResult] = field(default_factory=list)


# Gold contract specs
GOLD_PIP_VALUE = 0.01  # 1 pip = $0.01 in gold price
GOLD_VALUE_PER_LOT = 100  # 1 standard lot = 100 oz
GOLD_SPREAD_PIPS = 30  # ~$0.30 spread (conservative for prop firm)
GOLD_COMMISSION_PER_LOT = 7.0  # $7 round trip
GOLD_SLIPPAGE_PIPS = 5  # ~$0.05 slippage


class PropFirmSimulator:
    def __init__(self,
                 account_size=100_000,
                 profit_target_pct=8.0,
                 max_daily_dd_pct=5.0,
                 max_total_dd_pct=10.0,
                 phase="Phase 1",
                 spread_pips=GOLD_SPREAD_PIPS,
                 commission_per_lot=GOLD_COMMISSION_PER_LOT,
                 slippage_pips=GOLD_SLIPPAGE_PIPS,
                 max_days=30):

        self.initial_balance = account_size
        self.profit_target = account_size * profit_target_pct / 100
        self.max_daily_dd = account_size * max_daily_dd_pct / 100
        self.max_total_dd = account_size * max_total_dd_pct / 100
        self.phase = phase
        self.spread = spread_pips * GOLD_PIP_VALUE
        self.commission_per_lot = commission_per_lot
        self.slippage = slippage_pips * GOLD_PIP_VALUE
        self.max_days = max_days

        self.profit_target_pct = profit_target_pct
        self.max_daily_dd_pct = max_daily_dd_pct
        self.max_total_dd_pct = max_total_dd_pct

    def simulate(self, df, signals, lot_sizes, stop_losses, take_profits):
        """
        Run prop firm challenge simulation.

        Parameters:
        -----------
        df : DataFrame with OHLCV data
        signals : array of signals (1=buy, -1=sell, 0=hold)
        lot_sizes : array of position sizes in lots
        stop_losses : array of SL distances in price
        take_profits : array of TP distances in price

        Returns:
        --------
        ChallengeResult
        """
        balance = self.initial_balance
        peak_balance = balance
        day_start_balance = balance

        trades = []
        daily_results = []
        active_trade = None
        current_day = None
        trading_days = set()
        day_pnl = 0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        max_daily_dd_seen = 0
        max_total_dd_seen = 0
        intraday_peak = balance
        breached = False
        breach_reason = ""

        for i in range(len(df)):
            bar = df.iloc[i]
            bar_date = pd.Timestamp(bar.name).date() if hasattr(bar.name, 'date') else bar.name

            # New day check
            if bar_date != current_day:
                if current_day is not None:
                    # Record previous day
                    day_dd = (intraday_peak - balance) if intraday_peak > balance else 0
                    daily_results.append(DayResult(
                        date=current_day,
                        starting_balance=day_start_balance,
                        ending_balance=balance,
                        pnl=day_pnl,
                        pnl_pct=day_pnl / self.initial_balance * 100,
                        max_drawdown_intraday=day_dd,
                        num_trades=day_trades,
                        winning_trades=day_wins,
                        losing_trades=day_losses
                    ))

                current_day = bar_date
                day_start_balance = balance
                intraday_peak = balance
                day_pnl = 0
                day_trades = 0
                day_wins = 0
                day_losses = 0

            # Check if we have an active trade
            if active_trade is not None:
                trade_dir = active_trade['direction']
                entry_price = active_trade['entry_price']
                lots = active_trade['lots']
                sl = active_trade['sl']
                tp = active_trade['tp']
                entry_idx = active_trade['entry_idx']
                entry_time = active_trade['entry_time']

                # Check SL/TP hits using High/Low
                exit_price = None
                exit_reason = None

                if trade_dir == 1:  # Long
                    if bar['Low'] <= sl:
                        exit_price = sl - self.slippage  # Slippage against us
                        exit_reason = 'sl'
                    elif bar['High'] >= tp:
                        exit_price = tp - self.slippage  # Conservative
                        exit_reason = 'tp'
                elif trade_dir == -1:  # Short
                    if bar['High'] >= sl:
                        exit_price = sl + self.slippage
                        exit_reason = 'sl'
                    elif bar['Low'] <= tp:
                        exit_price = tp + self.slippage
                        exit_reason = 'tp'

                # Signal reversal or exit
                if exit_price is None and i < len(signals):
                    sig = signals[i]
                    if (sig == -trade_dir) or sig == 0:
                        # Exit on opposite signal or flat signal
                        if sig != 0 or (i - entry_idx > 2):  # Don't exit too fast on 0
                            exit_price = bar['Close']
                            if trade_dir == 1:
                                exit_price -= self.spread / 2 + self.slippage
                            else:
                                exit_price += self.spread / 2 + self.slippage
                            exit_reason = 'signal'

                if exit_price is not None:
                    # Calculate P&L
                    if trade_dir == 1:
                        pnl_per_oz = exit_price - entry_price
                    else:
                        pnl_per_oz = entry_price - exit_price

                    pnl_dollars = pnl_per_oz * lots * GOLD_VALUE_PER_LOT - self.commission_per_lot * lots
                    pnl_pips = pnl_per_oz / GOLD_PIP_VALUE

                    balance += pnl_dollars
                    day_pnl += pnl_dollars
                    day_trades += 1

                    if pnl_dollars > 0:
                        day_wins += 1
                    elif pnl_dollars < 0:
                        day_losses += 1

                    trading_days.add(bar_date)

                    trades.append(TradeResult(
                        entry_time=entry_time,
                        exit_time=bar.name,
                        direction=trade_dir,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        lot_size=lots,
                        pnl_dollars=pnl_dollars,
                        pnl_pips=pnl_pips,
                        duration_bars=i - entry_idx,
                        exit_reason=exit_reason
                    ))

                    active_trade = None

            # Update drawdown tracking
            if balance > peak_balance:
                peak_balance = balance
            if balance > intraday_peak:
                intraday_peak = balance

            total_dd = peak_balance - balance
            daily_dd = intraday_peak - balance

            if total_dd > max_total_dd_seen:
                max_total_dd_seen = total_dd
            if daily_dd > max_daily_dd_seen:
                max_daily_dd_seen = daily_dd

            # Check breaches
            if daily_dd >= self.max_daily_dd:
                breached = True
                breach_reason = "daily_dd_breach"
                break
            if total_dd >= self.max_total_dd:
                breached = True
                breach_reason = "total_dd_breach"
                break

            # Check if target hit
            total_profit = balance - self.initial_balance
            if total_profit >= self.profit_target:
                breach_reason = "target_hit"
                break

            # Open new trade if no active trade
            if active_trade is None and i < len(signals):
                sig = signals[i]
                if sig != 0:
                    lots = lot_sizes[i] if i < len(lot_sizes) else 0.1
                    if lots <= 0:
                        continue

                    sl_dist = stop_losses[i] if i < len(stop_losses) else 5.0
                    tp_dist = take_profits[i] if i < len(take_profits) else 10.0

                    if sig == 1:  # Buy
                        entry_price = bar['Close'] + self.spread / 2 + self.slippage
                        sl_price = entry_price - sl_dist
                        tp_price = entry_price + tp_dist
                    else:  # Sell
                        entry_price = bar['Close'] - self.spread / 2 - self.slippage
                        sl_price = entry_price + sl_dist
                        tp_price = entry_price - tp_dist

                    # Risk check: don't risk more than daily DD limit remaining
                    potential_loss = sl_dist * lots * GOLD_VALUE_PER_LOT + self.commission_per_lot * lots
                    remaining_daily_dd = self.max_daily_dd - (intraday_peak - balance)
                    remaining_total_dd = self.max_total_dd - (peak_balance - balance)
                    max_allowed_loss = min(remaining_daily_dd, remaining_total_dd) * 0.5

                    if potential_loss > max_allowed_loss and max_allowed_loss > 0:
                        # Reduce lot size
                        lots = max(0.01, max_allowed_loss / (sl_dist * GOLD_VALUE_PER_LOT + self.commission_per_lot))
                        lots = round(lots, 2)
                        if sig == 1:
                            sl_price = entry_price - sl_dist
                            tp_price = entry_price + tp_dist
                        else:
                            sl_price = entry_price + sl_dist
                            tp_price = entry_price - tp_dist

                    active_trade = {
                        'direction': sig,
                        'entry_price': entry_price,
                        'lots': lots,
                        'sl': sl_price,
                        'tp': tp_price,
                        'entry_idx': i,
                        'entry_time': bar.name
                    }
                    trading_days.add(bar_date)

        # Record last day
        if current_day is not None:
            day_dd = (intraday_peak - balance) if intraday_peak > balance else 0
            daily_results.append(DayResult(
                date=current_day,
                starting_balance=day_start_balance,
                ending_balance=balance,
                pnl=day_pnl,
                pnl_pct=day_pnl / self.initial_balance * 100,
                max_drawdown_intraday=day_dd,
                num_trades=day_trades,
                winning_trades=day_wins,
                losing_trades=day_losses
            ))

        # Calculate final metrics
        total_pnl = balance - self.initial_balance
        total_pnl_pct = total_pnl / self.initial_balance * 100

        winning_trades = [t for t in trades if t.pnl_dollars > 0]
        losing_trades = [t for t in trades if t.pnl_dollars < 0]
        win_rate = len(winning_trades) / max(len(trades), 1) * 100

        avg_win = np.mean([t.pnl_dollars for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl_dollars) for t in losing_trades]) if losing_trades else 0
        total_wins = sum(t.pnl_dollars for t in winning_trades)
        total_losses = sum(abs(t.pnl_dollars) for t in losing_trades)
        profit_factor = total_wins / max(total_losses, 1)

        num_trading_days = len(trading_days)
        passed = (not breached and total_pnl >= self.profit_target and num_trading_days >= 5)

        if breach_reason == "target_hit" and num_trading_days >= 5:
            passed = True
        elif breach_reason == "target_hit" and num_trading_days < 5:
            reason = f"target_hit_but_only_{num_trading_days}_days"
            passed = False
            breach_reason = reason

        return ChallengeResult(
            passed=passed,
            phase=self.phase,
            reason=breach_reason if breach_reason else "time_expired",
            days_taken=num_trading_days,
            final_balance=balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_daily_dd=max_daily_dd_seen,
            max_total_dd=max_total_dd_seen,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            daily_results=daily_results,
            trades=trades
        )


def run_both_phases(df, signals, lot_sizes, stop_losses, take_profits):
    """Run both Phase 1 and Phase 2 of a prop firm challenge."""
    # Phase 1: 8% target
    sim1 = PropFirmSimulator(profit_target_pct=8.0, phase="Phase 1")
    result1 = sim1.simulate(df, signals, lot_sizes, stop_losses, take_profits)

    if not result1.passed:
        return result1, None

    # Phase 2: 5% target (use remaining data after Phase 1 ended)
    # Find where Phase 1 ended
    if result1.trades:
        last_trade_time = result1.trades[-1].exit_time
        phase2_start = df.index.get_loc(last_trade_time) + 1 if last_trade_time in df.index else 0
    else:
        phase2_start = 0

    if phase2_start >= len(df):
        return result1, None

    df2 = df.iloc[phase2_start:]
    signals2 = signals[phase2_start:]
    lots2 = lot_sizes[phase2_start:]
    sl2 = stop_losses[phase2_start:]
    tp2 = take_profits[phase2_start:]

    sim2 = PropFirmSimulator(profit_target_pct=5.0, phase="Phase 2")
    result2 = sim2.simulate(df2, signals2, lots2, sl2, tp2)

    return result1, result2
