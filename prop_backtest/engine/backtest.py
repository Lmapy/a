"""BacktestEngine — the main simulation loop."""
from __future__ import annotations

import copy
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from prop_backtest.account.monitor import RuleMonitor
from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.firms.base import FirmRules
from prop_backtest.reporting.results import BacktestResult, RuleCheckResult, TradeRecord
from prop_backtest.reporting.stats import compute_statistics
from .broker import Fill, Signal, SimulatedBroker

if TYPE_CHECKING:
    from prop_backtest.strategy.base import Strategy


class BacktestEngine:
    """Runs a strategy against historical bars and enforces prop firm rules.

    Usage:
        engine = BacktestEngine(strategy, firm_rules, tier_name="100K", contract=ES)
        result = engine.run(bars)
    """

    def __init__(
        self,
        strategy: "Strategy",
        firm_rules: FirmRules,
        tier_name: str,
        contract: ContractSpec,
        commission_per_rt: float = 4.50,
        slippage_ticks: int = 0,
        fill_on: Literal["next_open", "current_close"] = "next_open",
    ) -> None:
        self.strategy = strategy
        self.firm_rules = firm_rules
        self.tier = firm_rules.get_tier(tier_name)
        self.contract = contract
        self.commission_per_rt = commission_per_rt
        self.slippage_ticks = slippage_ticks
        self.fill_on = fill_on

    def run(self, bars: list[BarData]) -> BacktestResult:
        """Simulate the full backtest.

        Args:
            bars: List of BarData, sorted ascending by timestamp.

        Returns:
            BacktestResult with full trade log, equity curve, and pass/fail verdict.
        """
        if not bars:
            raise ValueError("bars list is empty")

        state = AccountState(
            tier=self.tier,
            firm_rules=self.firm_rules,
            contract=self.contract,
        )
        broker = SimulatedBroker(
            commission_per_rt=self.commission_per_rt,
            slippage_ticks=self.slippage_ticks,
        )
        monitor = RuleMonitor()

        self.strategy.on_start(self.contract, self.tier)

        history: list[BarData] = []
        equity_curve: list[tuple[datetime, float]] = []
        fills: list[Fill] = []
        trade_records: list[TradeRecord] = []

        # Track open trade for building TradeRecord on close
        open_trade: Optional[dict] = None
        prev_date = None

        for bar_index, bar in enumerate(bars):
            # ── Step 1: EOD update when date changes ─────────────────────
            current_date = bar.date
            if prev_date is not None and current_date != prev_date:
                monitor.update_eod(state)
            state.current_date = current_date
            prev_date = current_date

            # ── Step 2: Execute pending order at this bar's open ──────────
            fill = broker.process_pending(bar, bar_index, state)
            if fill is not None:
                fills.append(fill)
                self.strategy.on_fill(fill)
                state.trading_days_active.add(current_date)

                if fill.action in ("buy", "sell") and open_trade is None:
                    open_trade = {
                        "trade_id": len(trade_records) + 1,
                        "entry_time": fill.timestamp,
                        "direction": fill.direction,
                        "contracts": fill.contracts,
                        "entry_price": fill.fill_price,
                        "entry_bar_index": bar_index,
                        "gross_pnl": 0.0,
                        "commission": fill.commission,
                    }
                elif fill.action == "close" and open_trade is not None:
                    mae, mfe = broker.get_excursion(state)
                    gross_pnl = fill.realized_pnl + fill.commission
                    net_pnl = fill.realized_pnl
                    trade_records.append(TradeRecord(
                        trade_id=open_trade["trade_id"],
                        entry_time=open_trade["entry_time"],
                        exit_time=fill.timestamp,
                        direction=open_trade["direction"],
                        contracts=open_trade["contracts"],
                        entry_price=open_trade["entry_price"],
                        exit_price=fill.fill_price,
                        gross_pnl=gross_pnl,
                        commission=fill.commission + open_trade["commission"],
                        net_pnl=net_pnl,
                        mae=mae,
                        mfe=mfe,
                    ))
                    open_trade = None

            # ── Step 3: Update open PnL from bar's close ──────────────────
            monitor.update_open_pnl(state, bar)
            broker.update_excursion(bar)

            # ── Step 4: Update intraday HWM ───────────────────────────────
            monitor.update_intraday_hwm(state)

            # ── Step 5: Check for violations ──────────────────────────────
            terminated = monitor.check_violations(state)
            equity_curve.append((bar.timestamp, state.equity))

            if terminated:
                # Force-close open position at bar's close
                if state.position_contracts != 0:
                    close_sig = Signal(action="close", contracts=0)
                    broker.submit(close_sig, bar_index)
                    # Execute immediately at close (exceptional path)
                    forced_fill = _force_close(state, broker, bar, bar_index)
                    if forced_fill is not None:
                        fills.append(forced_fill)
                        if open_trade is not None:
                            mae, mfe = broker.get_excursion(state)
                            gross_pnl = forced_fill.realized_pnl + forced_fill.commission
                            trade_records.append(TradeRecord(
                                trade_id=open_trade["trade_id"],
                                entry_time=open_trade["entry_time"],
                                exit_time=forced_fill.timestamp,
                                direction=open_trade["direction"],
                                contracts=open_trade["contracts"],
                                entry_price=open_trade["entry_price"],
                                exit_price=forced_fill.fill_price,
                                gross_pnl=gross_pnl,
                                commission=forced_fill.commission + open_trade["commission"],
                                net_pnl=forced_fill.realized_pnl,
                                mae=mae,
                                mfe=mfe,
                            ))
                            open_trade = None
                break

            # ── Step 6: Append bar to history & call strategy ─────────────
            history.append(bar)
            signal = self.strategy.on_bar(bar, history, state)
            if signal is not None and signal.action != "hold":
                broker.submit(signal, bar_index)

        self.strategy.on_end()

        # ── Build result ──────────────────────────────────────────────────
        rule_checks = _build_rule_checks(state)
        passed = all(r.passed for r in rule_checks)
        failure_reason = next((r.rule_name for r in rule_checks if not r.passed), None)

        stats = compute_statistics(
            trade_records,
            equity_curve,
            state.tier.starting_balance,
        )

        return BacktestResult(
            firm_name=self.firm_rules.firm_name,
            tier_name=self.tier.name,
            contract_symbol=self.contract.symbol,
            start_date=bars[0].date,
            end_date=bars[-1].date,
            starting_balance=state.tier.starting_balance,
            final_realized_balance=state.realized_balance,
            final_equity=state.equity,
            peak_equity=state.intraday_hwm,
            max_drawdown_dollars=state.intraday_hwm - min(eq for _, eq in equity_curve),
            trades=trade_records,
            equity_curve=equity_curve,
            rule_checks=rule_checks,
            passed=passed,
            failure_reason=failure_reason,
            stats=stats,
        )


def _force_close(
    state: AccountState,
    broker: SimulatedBroker,
    bar: BarData,
    bar_index: int,
) -> Optional[Fill]:
    """Close position at the current bar's close price (termination path)."""
    if state.position_contracts == 0:
        return None

    contracts = abs(state.position_contracts)
    is_short = state.position_contracts < 0
    fill_price = bar.close

    commission = broker.commission_per_rt * contracts
    realized_pnl = state.contract.pnl(
        state.avg_entry_price, fill_price, state.position_contracts, is_short=is_short
    )
    realized_pnl -= commission

    state.realized_balance += realized_pnl
    state.open_pnl = 0.0
    state.position_contracts = 0
    state.avg_entry_price = 0.0

    return Fill(
        bar_index=bar_index,
        timestamp=bar.timestamp,
        action="close",
        direction="long" if not is_short else "short",
        contracts=contracts,
        fill_price=fill_price,
        commission=commission,
        realized_pnl=realized_pnl + commission,
        net_pnl=realized_pnl,
        slippage_ticks=0,
    )


def _build_rule_checks(state: AccountState) -> list[RuleCheckResult]:
    tier = state.tier
    checks = []

    # Profit target
    actual_profit = state.realized_balance - tier.starting_balance
    checks.append(RuleCheckResult(
        rule_name="profit_target",
        passed=not state.breached_trailing_dd
               and not state.breached_daily_loss
               and state.hit_profit_target,
        value=actual_profit,
        threshold=tier.profit_target,
        detail=(
            f"Profit ${actual_profit:,.2f} vs target ${tier.profit_target:,.2f}"
        ),
    ))

    # Trailing drawdown (not breached)
    checks.append(RuleCheckResult(
        rule_name="trailing_drawdown",
        passed=not state.breached_trailing_dd,
        value=0.0 if state.breached_trailing_dd else 1.0,
        threshold=1.0,
        detail=(
            "BREACHED — account blown" if state.breached_trailing_dd else "OK — not breached"
        ),
    ))

    # Daily loss limit (not breached)
    checks.append(RuleCheckResult(
        rule_name="daily_loss_limit",
        passed=not state.breached_daily_loss,
        value=0.0 if state.breached_daily_loss else 1.0,
        threshold=1.0,
        detail=(
            "BREACHED — daily loss exceeded" if state.breached_daily_loss else "OK — not breached"
        ),
    ))

    # Minimum trading days
    actual_days = len(state.trading_days_active)
    checks.append(RuleCheckResult(
        rule_name="min_trading_days",
        passed=actual_days >= tier.min_trading_days,
        value=float(actual_days),
        threshold=float(tier.min_trading_days),
        detail=f"{actual_days} trading days vs minimum {tier.min_trading_days}",
    ))

    return checks
