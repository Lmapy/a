from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.backtest.challenge_sim import ChallengeSim, ChallengeResult
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import format_metrics
from src.backtest.monte_carlo import MonteCarloSimulator
from src.core.config import GlobalConfig, load_global_config
from src.core.journal import TradeJournal
from src.data.feed import CSVDataFeed, DataFeed, YFinanceDataFeed
from src.strategy.base import BaseStrategy
from src.strategy.breakout import BreakoutStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Main entry point. Wires all components together."""

    def __init__(self, config_dir: str | Path = "config"):
        self.config = load_global_config(config_dir)
        self.strategies = self._build_strategies()

    def _build_strategies(self) -> dict[str, BaseStrategy]:
        strategies: dict[str, BaseStrategy] = {}

        strat_configs = self.config.strategies

        # Only build strategies that are enabled in config (or default all if no config)
        build_defaults = len(strat_configs) == 0

        if "mean_reversion" in strat_configs or build_defaults:
            params = strat_configs["mean_reversion"].parameters if "mean_reversion" in strat_configs else {}
            strategies["mean_reversion"] = MeanReversionStrategy(params)

        if "trend_following" in strat_configs or build_defaults:
            params = strat_configs["trend_following"].parameters if "trend_following" in strat_configs else {}
            strategies["trend_following"] = TrendFollowingStrategy(params)

        if "breakout" in strat_configs or build_defaults:
            params = strat_configs["breakout"].parameters if "breakout" in strat_configs else {}
            strategies["breakout"] = BreakoutStrategy(params)

        return strategies

    def backtest(self, data: pd.DataFrame, instrument: str = "ES") -> BacktestResult:
        """Run a backtest on provided data."""
        engine = BacktestEngine(self.config, self.strategies)
        result = engine.run(data, instrument)

        print(format_metrics(result.metrics))
        return result

    def challenge_sim(self, data: pd.DataFrame, instrument: str = "ES") -> ChallengeResult:
        """Simulate a single prop firm challenge."""
        sim = ChallengeSim(self.config, self.strategies)
        result = sim.run(data, instrument)

        print(format_metrics(result.backtest_result.metrics))
        return result

    def monte_carlo(self, daily_pnls: list[float],
                    n_trials: int = 10000) -> None:
        """Run Monte Carlo pass rate estimation."""
        mc = MonteCarloSimulator(
            profit_target=self.config.firm.profit_target,
            drawdown_amount=self.config.firm.trailing_drawdown.initial_amount,
            n_trials=n_trials,
        )
        result = mc.run(daily_pnls)
        print(MonteCarloSimulator.format_result(result))

    def run_full_analysis(self, data: pd.DataFrame, instrument: str = "ES",
                          output_dir: str = "output") -> None:
        """Run backtest, challenge sim, and Monte Carlo in sequence."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. Backtest
        print("\n--- BACKTEST ---")
        bt_result = self.backtest(data, instrument)

        # Save trade journal
        journal = TradeJournal(output_path / "trades.csv")
        journal.add_trades(bt_result.trades)
        journal.save()

        # 2. Monte Carlo (if we have daily P&Ls)
        if bt_result.daily_pnls and len(bt_result.daily_pnls) >= 5:
            print("\n--- MONTE CARLO SIMULATION ---")
            self.monte_carlo(bt_result.daily_pnls)

        # 3. Challenge sim summary
        if bt_result.daily_pnls:
            print(f"\nChallenge outcome: {bt_result.status.value}")
            print(f"Days traded: {len(bt_result.daily_pnls)}")
            print(f"Final balance: ${bt_result.daily_balances[-1] if bt_result.daily_balances else 0:,.2f}")
