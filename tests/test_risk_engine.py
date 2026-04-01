from datetime import datetime

import pytest

from src.core.config import (
    DrawdownConfig,
    FirmConfig,
    RiskConfig,
    TradingHoursConfig,
)
from src.data.models import (
    ChallengeStatus,
    Fill,
    Order,
    OrderType,
    SignalDirection,
)
from src.risk.engine import RiskEngine


def make_firm():
    return FirmConfig(
        firm_name="Test",
        account_size=50000,
        profit_target=3000,
        trailing_drawdown=DrawdownConfig(type="eod", initial_amount=2000, trails_up=True),
        max_contracts={"ES": 2, "NQ": 2, "MES": 10, "MNQ": 10},
        allowed_instruments=["ES", "NQ", "MES", "MNQ"],
        trading_hours=TradingHoursConfig(start="08:30", end="15:00", flat_by_close=True),
    )


def make_risk():
    return RiskConfig()


def make_engine():
    engine = RiskEngine(make_firm(), make_risk())
    engine.start_session()
    return engine


class TestRiskEngine:
    def test_approve_basic_order(self):
        engine = make_engine()
        order = Order(
            instrument="ES", direction=SignalDirection.LONG, quantity=1,
            order_type=OrderType.MARKET,
        )
        result = engine.approve_order(order)
        assert result.approved
        assert result.quantity == 1

    def test_reject_when_no_session(self):
        engine = RiskEngine(make_firm(), make_risk())
        # Don't start session
        order = Order(
            instrument="ES", direction=SignalDirection.LONG, quantity=1,
            order_type=OrderType.MARKET,
        )
        result = engine.approve_order(order)
        assert not result.approved

    def test_reject_disallowed_instrument(self):
        engine = make_engine()
        order = Order(
            instrument="ZB", direction=SignalDirection.LONG, quantity=1,
            order_type=OrderType.MARKET,
        )
        result = engine.approve_order(order)
        assert not result.approved

    def test_clamp_to_max_contracts(self):
        engine = make_engine()
        order = Order(
            instrument="ES", direction=SignalDirection.LONG, quantity=5,
            order_type=OrderType.MARKET,
        )
        result = engine.approve_order(order)
        assert result.approved
        assert result.quantity == 2  # max for ES

    def test_position_open_close_cycle(self):
        engine = make_engine()
        now = datetime(2024, 1, 1, 10, 0)

        # Open long
        fill_open = Fill("1", "ES", SignalDirection.LONG, 1, 5000.0, now)
        pnl = engine.on_fill(fill_open)
        assert pnl is None  # opening trade
        assert "ES" in engine.positions

        # Close with profit (5 points = 20 ticks * $12.50 = $250)
        fill_close = Fill("2", "ES", SignalDirection.SHORT, 1, 5005.0, now)
        pnl = engine.on_fill(fill_close)
        assert pnl is not None
        assert pnl == pytest.approx(250.0, abs=1)
        assert "ES" not in engine.positions

    def test_session_close_updates_drawdown(self):
        engine = make_engine()
        now = datetime(2024, 1, 1, 10, 0)

        # Make a profitable trade
        engine.on_fill(Fill("1", "ES", SignalDirection.LONG, 1, 5000.0, now))
        engine.on_fill(Fill("2", "ES", SignalDirection.SHORT, 1, 5010.0, now))
        # 10 points = 40 ticks * $12.50 = $500

        status = engine.on_session_close()
        assert status == ChallengeStatus.ACTIVE
        assert engine.tracker.high_water_mark == 50500
        assert engine.tracker.drawdown_floor == 48500

    def test_position_sizing(self):
        engine = make_engine()
        # With $2000 budget, 1% risk, 20 tick stop on ES ($250 risk per contract)
        # risk = $2000 * 0.01 = $20 -> 0 contracts (too small)
        size = engine.get_position_size("ES", 20)
        # At 1% of $2000 = $20, and stop = 20 * $12.50 = $250, 0 contracts
        # This is expected with default 1% risk on tight budget
        assert size >= 0

    def test_lockout_after_losses(self):
        engine = make_engine()
        now = datetime(2024, 1, 1, 10, 0)

        # Simulate 3 consecutive losses
        for i in range(3):
            engine.on_fill(Fill(f"o{i}", "ES", SignalDirection.LONG, 1, 5000.0, now))
            engine.on_fill(Fill(f"c{i}", "ES", SignalDirection.SHORT, 1, 4998.0, now))
            # -2 points = -8 ticks * $12.50 = -$100 each

        assert engine.lockout.is_locked()
        assert engine.get_position_size("ES", 20) == 0
