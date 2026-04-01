from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    pass_rate: float
    avg_days_to_pass: float
    risk_of_ruin: float
    median_days_to_pass: float
    pass_rate_ci_lower: float
    pass_rate_ci_upper: float
    avg_profit_when_passed: float
    avg_loss_when_blown: float
    optimal_daily_target: float
    days_to_pass_distribution: list[int]
    final_pnl_distribution: list[float]


class MonteCarloSimulator:
    """Estimates challenge pass rate using Monte Carlo simulation.

    Uses daily P&L resampling with block bootstrap to preserve
    serial correlation (trending days cluster, volatility clusters).

    For each trial:
        1. Block-bootstrap sample daily P&Ls
        2. Simulate EOD trailing drawdown rules
        3. Check if profit target hit before drawdown floor hit
    """

    def __init__(self, profit_target: float = 3000.0, drawdown_amount: float = 2000.0,
                 n_trials: int = 10000, max_days: int = 60, block_size: int = 3):
        self.profit_target = profit_target
        self.drawdown_amount = drawdown_amount
        self.n_trials = n_trials
        self.max_days = max_days
        self.block_size = block_size

    def run(self, daily_pnls: list[float] | np.ndarray) -> MonteCarloResult:
        """Run Monte Carlo simulation on historical daily P&L data."""
        pnls = np.array(daily_pnls, dtype=float)

        if len(pnls) < 5:
            raise ValueError(f"Need at least 5 daily P&Ls, got {len(pnls)}")

        logger.info(
            f"Running Monte Carlo: {self.n_trials} trials, "
            f"target=${self.profit_target}, DD=${self.drawdown_amount}, "
            f"max_days={self.max_days}, block_size={self.block_size}"
        )
        logger.info(
            f"Daily P&L stats: mean=${np.mean(pnls):.2f}, "
            f"std=${np.std(pnls):.2f}, "
            f"min=${np.min(pnls):.2f}, max=${np.max(pnls):.2f}"
        )

        passes = 0
        blown = 0
        days_to_pass: list[int] = []
        profit_when_passed: list[float] = []
        loss_when_blown: list[float] = []
        final_pnls: list[float] = []

        for trial in range(self.n_trials):
            sampled = self._block_bootstrap(pnls, self.max_days)
            result = self._simulate_challenge(sampled)

            final_pnls.append(result["final_pnl"])

            if result["passed"]:
                passes += 1
                days_to_pass.append(result["days"])
                profit_when_passed.append(result["final_pnl"])
            elif result["blown"]:
                blown += 1
                loss_when_blown.append(result["final_pnl"])

        pass_rate = passes / self.n_trials

        # Bootstrap confidence interval for pass rate
        ci_lower, ci_upper = self._bootstrap_ci(pnls, n_bootstrap=500)

        # Find optimal daily target
        optimal_target = self._find_optimal_daily_target(pnls)

        result = MonteCarloResult(
            pass_rate=pass_rate,
            avg_days_to_pass=float(np.mean(days_to_pass)) if days_to_pass else float("inf"),
            risk_of_ruin=blown / self.n_trials,
            median_days_to_pass=float(np.median(days_to_pass)) if days_to_pass else float("inf"),
            pass_rate_ci_lower=ci_lower,
            pass_rate_ci_upper=ci_upper,
            avg_profit_when_passed=float(np.mean(profit_when_passed)) if profit_when_passed else 0,
            avg_loss_when_blown=float(np.mean(loss_when_blown)) if loss_when_blown else 0,
            optimal_daily_target=optimal_target,
            days_to_pass_distribution=days_to_pass,
            final_pnl_distribution=final_pnls,
        )

        logger.info(
            f"Monte Carlo complete: pass_rate={pass_rate:.1%}, "
            f"avg_days={result.avg_days_to_pass:.1f}, "
            f"risk_of_ruin={result.risk_of_ruin:.1%}, "
            f"CI=[{ci_lower:.1%}, {ci_upper:.1%}]"
        )

        return result

    def _block_bootstrap(self, pnls: np.ndarray, n_days: int) -> np.ndarray:
        """Sample n_days of P&L using block bootstrap to preserve serial correlation."""
        n = len(pnls)
        result = []

        while len(result) < n_days:
            # Random start index for a block
            start = np.random.randint(0, max(1, n - self.block_size + 1))
            block = pnls[start:start + self.block_size]
            result.extend(block.tolist())

        return np.array(result[:n_days])

    def _simulate_challenge(self, daily_pnls: np.ndarray) -> dict:
        """Simulate one challenge attempt with EOD trailing drawdown."""
        balance = 0.0  # relative to starting balance
        hwm = 0.0
        floor = -self.drawdown_amount

        for day_idx, day_pnl in enumerate(daily_pnls):
            balance += day_pnl

            # Update HWM (EOD trailing)
            if balance > hwm:
                hwm = balance
                floor = hwm - self.drawdown_amount

            # Check blown
            if balance <= floor:
                return {"passed": False, "blown": True, "days": day_idx + 1, "final_pnl": balance}

            # Check passed
            if balance >= self.profit_target:
                return {"passed": True, "blown": False, "days": day_idx + 1, "final_pnl": balance}

        # Didn't finish within max days
        return {"passed": False, "blown": False, "days": len(daily_pnls), "final_pnl": balance}

    def _bootstrap_ci(self, pnls: np.ndarray, n_bootstrap: int = 500,
                      confidence: float = 0.95) -> tuple[float, float]:
        """Calculate bootstrap confidence interval for pass rate."""
        boot_rates = []
        alpha = (1 - confidence) / 2

        for _ in range(n_bootstrap):
            # Resample the source P&Ls
            boot_pnls = pnls[np.random.randint(0, len(pnls), size=len(pnls))]

            # Run a quick MC with fewer trials
            passes = 0
            quick_trials = 500
            for _ in range(quick_trials):
                sampled = self._block_bootstrap(boot_pnls, self.max_days)
                result = self._simulate_challenge(sampled)
                if result["passed"]:
                    passes += 1
            boot_rates.append(passes / quick_trials)

        return float(np.percentile(boot_rates, alpha * 100)), float(np.percentile(boot_rates, (1 - alpha) * 100))

    def _find_optimal_daily_target(self, pnls: np.ndarray) -> float:
        """Find the daily P&L target that maximizes pass rate.

        Tests different daily stop-gain amounts to see which produces
        the best pass rate. The idea: if you stop trading after making
        $X/day, what $X maximizes your challenge pass probability?
        """
        best_rate = 0
        best_target = 0
        mean_pnl = float(np.mean(pnls))

        # Test targets from small to large
        targets = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]

        for target in targets:
            # Cap positive days at target
            capped = np.clip(pnls, None, target)

            passes = 0
            trials = 1000
            for _ in range(trials):
                sampled = self._block_bootstrap(capped, self.max_days)
                result = self._simulate_challenge(sampled)
                if result["passed"]:
                    passes += 1

            rate = passes / trials
            if rate > best_rate:
                best_rate = rate
                best_target = target

        return float(best_target)

    @staticmethod
    def format_result(result: MonteCarloResult) -> str:
        lines = [
            "=" * 60,
            "MONTE CARLO SIMULATION RESULTS",
            "=" * 60,
            f"Pass Rate:           {result.pass_rate:.1%}  "
            f"[{result.pass_rate_ci_lower:.1%} - {result.pass_rate_ci_upper:.1%}] 95% CI",
            f"Risk of Ruin:        {result.risk_of_ruin:.1%}",
            f"Avg Days to Pass:    {result.avg_days_to_pass:.1f}",
            f"Median Days to Pass: {result.median_days_to_pass:.1f}",
            f"Avg Profit (passed): ${result.avg_profit_when_passed:,.2f}",
            f"Avg Loss (blown):    ${result.avg_loss_when_blown:,.2f}",
            f"Optimal Daily Target:${result.optimal_daily_target:,.0f}",
            "=" * 60,
        ]
        return "\n".join(lines)
