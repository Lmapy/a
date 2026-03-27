"""Tests for AccountState and RuleMonitor."""
import pytest
from datetime import date, datetime

from prop_backtest.account.monitor import RuleMonitor
from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import get_contract
from prop_backtest.data.loader import BarData
from prop_backtest.firms import TOPSTEP_RULES, MFF_RULES


def make_state(firm=TOPSTEP_RULES, tier_name="50K"):
    contract = get_contract("ES")
    return AccountState(tier=firm.get_tier(tier_name), firm_rules=firm, contract=contract)


def make_bar(close=5000.0, open_=5000.0, high=5010.0, low=4990.0, dt=None):
    if dt is None:
        dt = datetime(2024, 1, 2, 9, 30)
    return BarData(
        timestamp=dt,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000,
        contract=get_contract("ES"),
    )


# ── AccountState basic ────────────────────────────────────────────────────────

def test_initial_state():
    state = make_state()
    assert state.realized_balance == 50_000
    assert state.equity == 50_000
    assert state.open_pnl == 0.0
    assert state.position_contracts == 0
    assert state.drawdown_floor == 48_000   # 50000 - 2000


def test_equity_property():
    state = make_state()
    state.realized_balance = 51_000
    state.open_pnl = -500
    assert state.equity == 50_500


def test_current_day_pnl():
    state = make_state()
    state.realized_balance = 50_500
    assert state.current_day_pnl == 500.0


# ── RuleMonitor ───────────────────────────────────────────────────────────────

class TestRuleMonitor:
    def setup_method(self):
        self.monitor = RuleMonitor()

    def test_open_pnl_long(self):
        state = make_state()
        state.position_contracts = 1
        state.avg_entry_price = 5000.0
        bar = make_bar(close=5004.0)  # +4 pts = 16 ticks = $200
        self.monitor.update_open_pnl(state, bar)
        assert state.open_pnl == pytest.approx(200.0)

    def test_open_pnl_short(self):
        state = make_state()
        state.position_contracts = -1
        state.avg_entry_price = 5000.0
        bar = make_bar(close=4996.0)  # short +4 pts = $200
        self.monitor.update_open_pnl(state, bar)
        assert state.open_pnl == pytest.approx(200.0)

    def test_open_pnl_flat(self):
        state = make_state()
        bar = make_bar(close=5100.0)
        self.monitor.update_open_pnl(state, bar)
        assert state.open_pnl == 0.0

    def test_intraday_hwm_updates(self):
        state = make_state()
        state.open_pnl = 1000.0   # equity = 51000
        self.monitor.update_intraday_hwm(state)
        assert state.intraday_hwm == 51_000.0
        # Floor: min(51000-2000, 50000) = 49000... wait, rule:
        # raw_floor = 51000 - 2000 = 49000; min(49000, 50000) = 49000
        assert state.drawdown_floor == 49_000.0

    def test_lock_to_breakeven(self):
        """Floor should cap at starting_balance once profitable."""
        state = make_state()
        # Equity must exceed start + DD to trigger lock-to-breakeven
        # TopStep 50K: start=50000, DD=2000. Lock when HWM > 50000+2000 = 52000
        state.open_pnl = 3_000.0   # equity = 53000 -> HWM = 53000
        self.monitor.update_intraday_hwm(state)
        assert state.intraday_hwm == 53_000.0
        # raw_floor = 53000 - 2000 = 51000; min(51000, 50000) = 50000
        assert state.drawdown_floor == 50_000.0

    def test_trailing_dd_breach(self):
        state = make_state()
        state.equity  # 50000, floor 48000
        # Drop equity below floor
        state.realized_balance = 47_000
        state.open_pnl = 0.0
        terminated = self.monitor.check_violations(state)
        assert terminated is True
        assert state.breached_trailing_dd is True
        assert state.is_terminated is True

    def test_daily_loss_breach(self):
        state = make_state()  # daily limit = 1000
        # Simulate a $1001 intraday loss
        state.open_pnl = -1_001.0
        terminated = self.monitor.check_violations(state)
        assert terminated is True
        assert state.breached_daily_loss is True

    def test_profit_target(self):
        state = make_state()  # target = 3000
        state.realized_balance = 53_001.0
        terminated = self.monitor.check_violations(state)
        assert terminated is True
        assert state.hit_profit_target is True

    def test_eod_update_trailing_eod(self):
        """EOD HWM should update at end of day for TRAILING_EOD firms."""
        state = AccountState(
            tier=MFF_RULES.get_tier("50K"),
            firm_rules=MFF_RULES,
            contract=get_contract("ES"),
        )
        state.realized_balance = 51_000.0
        self.monitor.update_eod(state)
        assert state.eod_hwm == 51_000.0
        # raw_floor = 51000 - 2500 = 48500; min(48500, 50000) = 48500
        assert state.drawdown_floor == 48_500.0
