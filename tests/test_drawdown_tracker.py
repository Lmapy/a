import pytest

from src.data.models import ChallengeStatus
from src.risk.drawdown_tracker import EODTrailingDrawdownTracker


def make_tracker(balance=50000, dd=2000, target=3000):
    return EODTrailingDrawdownTracker(balance, dd, target)


class TestEODTrailingDrawdown:
    def test_initial_state(self):
        t = make_tracker()
        assert t.current_balance == 50000
        assert t.high_water_mark == 50000
        assert t.drawdown_floor == 48000
        assert t.remaining_budget == 2000
        assert t.profit_so_far == 0
        assert t.status == ChallengeStatus.ACTIVE

    def test_profitable_day_raises_hwm(self):
        t = make_tracker()
        t.start_session()
        t.on_realized_pnl(500)
        status = t.on_session_close()

        assert status == ChallengeStatus.ACTIVE
        assert t.high_water_mark == 50500
        assert t.drawdown_floor == 48500  # trails up
        assert t.remaining_budget == 2000  # budget stays the same relative to floor

    def test_losing_day_does_not_lower_hwm(self):
        t = make_tracker()
        t.start_session()
        t.on_realized_pnl(-300)
        status = t.on_session_close()

        assert status == ChallengeStatus.ACTIVE
        assert t.high_water_mark == 50000  # unchanged
        assert t.drawdown_floor == 48000  # unchanged
        assert t.remaining_budget == 1700  # balance dropped

    def test_blow_account(self):
        t = make_tracker()
        t.start_session()
        t.on_realized_pnl(-2000)
        status = t.on_session_close()

        assert status == ChallengeStatus.BLOWN
        assert t.current_balance == 48000
        assert t.current_balance <= t.drawdown_floor

    def test_pass_challenge(self):
        t = make_tracker()
        t.start_session()
        t.on_realized_pnl(3000)
        status = t.on_session_close()

        assert status == ChallengeStatus.PASSED
        assert t.profit_so_far == 3000

    def test_multi_day_sequence(self):
        t = make_tracker()

        # Day 1: win $800
        t.start_session()
        t.on_realized_pnl(800)
        t.on_session_close()
        assert t.high_water_mark == 50800
        assert t.drawdown_floor == 48800

        # Day 2: lose $500
        t.start_session()
        t.on_realized_pnl(-500)
        t.on_session_close()
        assert t.high_water_mark == 50800  # no change
        assert t.drawdown_floor == 48800  # no change
        assert t.current_balance == 50300
        assert t.remaining_budget == 1500

        # Day 3: win $700
        t.start_session()
        t.on_realized_pnl(700)
        t.on_session_close()
        assert t.high_water_mark == 51000  # new high
        assert t.drawdown_floor == 49000

    def test_hwm_trailing_then_blow(self):
        """Win big, then HWM rises, then lose back to floor."""
        t = make_tracker()

        # Day 1: win $1500 -> HWM = 51500, floor = 49500
        t.start_session()
        t.on_realized_pnl(1500)
        t.on_session_close()
        assert t.drawdown_floor == 49500

        # Day 2: lose $2000 -> balance = 49500, exactly at floor
        t.start_session()
        t.on_realized_pnl(-2000)
        status = t.on_session_close()
        assert status == ChallengeStatus.BLOWN

    def test_exactly_at_profit_target(self):
        t = make_tracker()
        t.start_session()
        t.on_realized_pnl(3000)
        status = t.on_session_close()
        assert status == ChallengeStatus.PASSED

    def test_daily_pnl_tracking(self):
        t = make_tracker()
        t.start_session()
        t.on_realized_pnl(200)
        t.on_realized_pnl(-50)
        assert t.daily_pnl == 150

    def test_unrealized_does_not_affect_balance(self):
        t = make_tracker()
        t.start_session()
        t.update_unrealized(-1500)  # big unrealized loss
        assert t.current_balance == 50000  # unchanged
        assert t.unrealized_pnl == -1500

    def test_gradual_pass(self):
        """Simulate passing over multiple days."""
        t = make_tracker()
        daily_gains = [400, 300, -100, 500, 200, 600, 300, 400, 500]

        for gain in daily_gains:
            t.start_session()
            t.on_realized_pnl(gain)
            status = t.on_session_close()
            if status != ChallengeStatus.ACTIVE:
                break

        assert status == ChallengeStatus.PASSED
        assert t.profit_so_far >= 3000
