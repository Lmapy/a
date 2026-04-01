from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import compute_metrics, format_metrics
from src.core.config import GlobalConfig
from src.data.models import ChallengeStatus
from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class ChallengeResult:
    passed: bool
    days_taken: int
    final_balance: float
    profit: float
    max_drawdown: float
    daily_pnls: list[float]
    backtest_result: BacktestResult


class ChallengeSim:
    """Simulates a complete prop firm challenge attempt.

    Wraps BacktestEngine and interprets results through the lens
    of prop firm challenge rules.
    """

    def __init__(self, config: GlobalConfig, strategies: dict[str, BaseStrategy]):
        self.config = config
        self.engine = BacktestEngine(config, strategies)

    def run(self, data: pd.DataFrame, instrument: str = "ES") -> ChallengeResult:
        """Run one complete challenge simulation."""
        logger.info(
            f"Starting challenge simulation: "
            f"{self.config.firm.firm_name} {self.config.firm.account_name}, "
            f"target=${self.config.firm.profit_target}, "
            f"drawdown=${self.config.firm.trailing_drawdown.initial_amount}"
        )

        bt = self.engine.run(data, instrument)

        passed = bt.status == ChallengeStatus.PASSED
        profit = bt.final_balance - self.config.firm.account_size

        result = ChallengeResult(
            passed=passed,
            days_taken=bt.days_traded,
            final_balance=bt.final_balance,
            profit=profit,
            max_drawdown=bt.metrics.get("max_drawdown", 0),
            daily_pnls=bt.daily_pnls,
            backtest_result=bt,
        )

        status_str = "PASSED" if passed else ("BLOWN" if bt.status == ChallengeStatus.BLOWN else "INCOMPLETE")
        logger.info(
            f"Challenge {status_str}: "
            f"{bt.days_traded} days, "
            f"P&L=${profit:+,.2f}, "
            f"final balance=${bt.final_balance:,.2f}"
        )

        return result

    def run_multiple(self, data: pd.DataFrame, instrument: str = "ES",
                     n_runs: int = 1, window_days: int = 60) -> list[ChallengeResult]:
        """Run multiple challenge simulations using rolling windows of data.

        Splits data into overlapping windows to simulate multiple challenge attempts
        on different market conditions.
        """
        results = []
        daily_groups = list(data.groupby(data.index.date))

        if len(daily_groups) < window_days:
            # Not enough data for multiple windows, just run once
            return [self.run(data, instrument)]

        step = max(1, window_days // 4)  # overlap by 75%

        for start_idx in range(0, len(daily_groups) - window_days + 1, step):
            if len(results) >= n_runs:
                break

            end_idx = start_idx + window_days
            window_dates = [d for d, _ in daily_groups[start_idx:end_idx]]
            start_date = window_dates[0]
            end_date = window_dates[-1]

            window_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
            if len(window_data) < 100:  # need minimum bars
                continue

            result = self.run(window_data, instrument)
            results.append(result)

        return results

    @staticmethod
    def summarize(results: list[ChallengeResult]) -> str:
        """Summarize multiple challenge results."""
        if not results:
            return "No results."

        n = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = passed / n

        avg_days = sum(r.days_taken for r in results) / n
        avg_profit = sum(r.profit for r in results) / n
        passed_days = [r.days_taken for r in results if r.passed]
        avg_pass_days = sum(passed_days) / len(passed_days) if passed_days else 0

        lines = [
            "=" * 60,
            "CHALLENGE SIMULATION SUMMARY",
            "=" * 60,
            f"Total Attempts:    {n}",
            f"Passed:            {passed} ({pass_rate:.1%})",
            f"Failed:            {n - passed}",
            f"Avg Days:          {avg_days:.1f}",
            f"Avg Days (passed): {avg_pass_days:.1f}",
            f"Avg Profit:        ${avg_profit:+,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)
