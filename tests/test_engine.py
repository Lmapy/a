"""Integration tests for BacktestEngine."""
import pytest
from datetime import datetime, date, timedelta

from prop_backtest.contracts.specs import get_contract
from prop_backtest.data.loader import BarData
from prop_backtest.engine.backtest import BacktestEngine
from prop_backtest.engine.broker import Signal
from prop_backtest.firms import TOPSTEP_RULES
from prop_backtest.strategy.base import FunctionStrategy, Strategy


# ── Helpers ───────────────────────────────────────────────────────────────────

ES = get_contract("ES")
TIER = TOPSTEP_RULES.get_tier("50K")


def make_bars(n: int, start_price: float = 5000.0, delta: float = 1.0) -> list[BarData]:
    """Generate n daily bars with close price increasing by delta each bar."""
    bars = []
    base = datetime(2024, 1, 2, 9, 30)
    for i in range(n):
        close = start_price + i * delta
        bars.append(BarData(
            timestamp=base + timedelta(days=i),
            open=close - 0.5,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=1000,
            contract=ES,
        ))
    return bars


def make_engine(strategy, commission=0.0, slippage=0):
    return BacktestEngine(
        strategy=strategy,
        firm_rules=TOPSTEP_RULES,
        tier_name="50K",
        contract=ES,
        commission_per_rt=commission,
        slippage_ticks=slippage,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_hold_strategy_no_trades():
    """A strategy that always holds should produce no trades."""
    hold = FunctionStrategy(lambda b, h, a: Signal("hold"))
    engine = make_engine(hold)
    result = engine.run(make_bars(30))
    assert result.stats["total_trades"] == 0
    assert result.final_realized_balance == 50_000.0


def test_buy_and_hold_wins():
    """Buying on first bar and selling at end should produce profit on rising bars."""
    class BuyOnce(Strategy):
        def __init__(self):
            self._bought = False
            self._bars_seen = 0

        def on_bar(self, bar, history, account):
            self._bars_seen += 1
            if not self._bought and len(history) >= 2:
                self._bought = True
                return Signal("buy", contracts=1)
            if self._bars_seen == 29:
                return Signal("close")
            return Signal("hold")

    engine = make_engine(BuyOnce(), commission=0.0)
    result = engine.run(make_bars(30, delta=2.0))
    # Should have positive net PnL
    assert result.final_realized_balance > 50_000.0
    assert result.stats["total_trades"] >= 1


def test_losing_strategy_terminates_engine():
    """A strategy that keeps losing should breach a rule and terminate early."""
    class AlwaysBuy(Strategy):
        def on_bar(self, bar, history, account):
            if account.position_contracts == 0:
                return Signal("buy", contracts=1)
            return Signal("hold")

    bars = make_bars(100, start_price=5000.0, delta=-5.0)  # price falls $5/bar
    engine = make_engine(AlwaysBuy(), commission=0.0)
    result = engine.run(bars)
    # Should terminate before all bars (daily loss or trailing DD breached)
    assert result.stats["trading_days"] < 100
    # At least one rule must be failed (daily_loss_limit or trailing_drawdown)
    failed_rules = [r for r in result.rule_checks if not r.passed]
    assert len(failed_rules) >= 1


def test_trailing_dd_breach_direct():
    """Directly verify trailing drawdown breach via account state."""
    from prop_backtest.account.monitor import RuleMonitor
    from prop_backtest.account.state import AccountState

    state = AccountState(
        tier=TOPSTEP_RULES.get_tier("50K"),
        firm_rules=TOPSTEP_RULES,
        contract=ES,
    )
    monitor = RuleMonitor()
    # Drop equity below floor (floor = 48000 at start)
    state.realized_balance = 47_500.0
    state.open_pnl = 0.0
    terminated = monitor.check_violations(state)
    assert terminated is True
    assert state.breached_trailing_dd is True


def test_profit_target_terminates_engine():
    """A profitable strategy should terminate when profit target is hit."""
    class CloseProfit(Strategy):
        """Opens long on bar 1, closes when equity nears target."""
        def on_bar(self, bar, history, account):
            if account.position_contracts == 0 and len(history) >= 2:
                return Signal("buy", contracts=1)
            # Close when equity is above profit target — realized balance then passes
            if account.equity >= 53_100 and account.position_contracts != 0:
                return Signal("close")
            return Signal("hold")

    # Rising bars: 1 ES contract * $50/pt * 1pt/bar = $50/bar; $3000 needs 60 bars
    bars = make_bars(200, start_price=5000.0, delta=1.0)
    engine = make_engine(CloseProfit(), commission=0.0)
    result = engine.run(bars)
    # After close, realized_balance should be >= 53000 → profit_target passes
    pt_check = next(r for r in result.rule_checks if r.rule_name == "profit_target")
    assert pt_check.value >= 3_000.0, (
        f"Expected profit >=3000, got {pt_check.value}. "
        f"Final realized: {result.final_realized_balance}"
    )


def test_rule_checks_all_present():
    hold = FunctionStrategy(lambda b, h, a: Signal("hold"))
    engine = make_engine(hold)
    result = engine.run(make_bars(15))
    rule_names = {r.rule_name for r in result.rule_checks}
    assert "profit_target" in rule_names
    assert "trailing_drawdown" in rule_names
    assert "daily_loss_limit" in rule_names
    assert "min_trading_days" in rule_names


def test_equity_curve_length():
    hold = FunctionStrategy(lambda b, h, a: Signal("hold"))
    engine = make_engine(hold)
    bars = make_bars(20)
    result = engine.run(bars)
    # Equity curve should have at most len(bars) entries
    assert len(result.equity_curve) <= len(bars)
    assert len(result.equity_curve) > 0


def test_min_trading_days_fail():
    """Should fail min_trading_days if no trades are made."""
    hold = FunctionStrategy(lambda b, h, a: Signal("hold"))
    engine = make_engine(hold)
    result = engine.run(make_bars(5))
    min_days_check = next(r for r in result.rule_checks if r.rule_name == "min_trading_days")
    # No trades were made → 0 active days
    assert min_days_check.value == 0.0
    assert min_days_check.passed is False


def test_empty_bars_raises():
    hold = FunctionStrategy(lambda b, h, a: Signal("hold"))
    engine = make_engine(hold)
    with pytest.raises(ValueError, match="bars list is empty"):
        engine.run([])
