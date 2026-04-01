import numpy as np
import pytest

from src.backtest.monte_carlo import MonteCarloSimulator


class TestMonteCarlo:
    def test_obvious_pass(self):
        """With consistently profitable daily P&Ls, pass rate should be very high."""
        # Average $200/day with small variance -> should pass $3000 target easily
        daily_pnls = list(np.random.normal(200, 50, 100))
        mc = MonteCarloSimulator(
            profit_target=3000, drawdown_amount=2000,
            n_trials=1000, max_days=60,
        )
        result = mc.run(daily_pnls)
        assert result.pass_rate > 0.5  # should be very high

    def test_obvious_fail(self):
        """With consistently losing daily P&Ls, pass rate should be near zero."""
        daily_pnls = list(np.random.normal(-200, 50, 100))
        mc = MonteCarloSimulator(
            profit_target=3000, drawdown_amount=2000,
            n_trials=1000, max_days=60,
        )
        result = mc.run(daily_pnls)
        assert result.pass_rate < 0.1

    def test_result_fields(self):
        """Check that all result fields are populated."""
        daily_pnls = list(np.random.normal(150, 80, 100))
        mc = MonteCarloSimulator(n_trials=500, max_days=60)
        result = mc.run(daily_pnls)

        assert 0 <= result.pass_rate <= 1
        assert 0 <= result.risk_of_ruin <= 1
        assert result.pass_rate_ci_lower <= result.pass_rate_ci_upper
        assert result.optimal_daily_target >= 0
        assert isinstance(result.final_pnl_distribution, list)

    def test_block_bootstrap_preserves_length(self):
        """Block bootstrap should return exactly n_days values."""
        mc = MonteCarloSimulator()
        pnls = np.array([100, -50, 200, -100, 150, -75, 300, -200, 50, 100])
        sampled = mc._block_bootstrap(pnls, 20)
        assert len(sampled) == 20

    def test_eod_drawdown_simulation(self):
        """Verify the challenge simulation logic directly."""
        mc = MonteCarloSimulator(profit_target=3000, drawdown_amount=2000)

        # Scenario: steady wins -> should pass
        result = mc._simulate_challenge(np.array([300] * 10))
        assert result["passed"]
        assert result["days"] == 10  # 300 * 10 = 3000

        # Scenario: immediate big loss -> should blow
        result = mc._simulate_challenge(np.array([-2000]))
        assert result["blown"]

        # Scenario: win then give back -> blow due to trailing
        # Win $1500 (HWM = 1500, floor = -500), then lose $2000 -> balance = -500 = floor -> blown
        result = mc._simulate_challenge(np.array([1500, -2000]))
        assert result["blown"]

    def test_format_result(self):
        daily_pnls = list(np.random.normal(100, 80, 50))
        mc = MonteCarloSimulator(n_trials=200, max_days=30)
        result = mc.run(daily_pnls)
        formatted = MonteCarloSimulator.format_result(result)
        assert "Pass Rate" in formatted
        assert "Risk of Ruin" in formatted

    def test_too_few_pnls_raises(self):
        mc = MonteCarloSimulator()
        with pytest.raises(ValueError):
            mc.run([100, -50])
