"""Walk-forward optimizer.

Splits the bar series into rolling in-sample / out-of-sample windows.
Optimises strategy parameters on in-sample, then evaluates on out-of-sample.
Stitches OOS results together to produce an unbiased equity curve.

Usage:
    from prop_backtest.analysis.walk_forward import WalkForwardOptimizer
    from prop_backtest.strategy.examples.sma_crossover import SMACrossover

    def strategy_factory(params):
        return SMACrossover(fast=params["fast"], slow=params["slow"])

    param_grid = [
        {"fast": f, "slow": s}
        for f in [3, 5, 9]
        for s in [10, 15, 21]
        if f < s
    ]

    wfo = WalkForwardOptimizer(
        firm_rules=TOPSTEP_RULES,
        tier_name="100K",
        contract=ES,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        is_window=120,   # bars
        oos_window=30,   # bars
        step=30,         # bars to advance each fold
    )
    wfo_result = wfo.run(bars)
    wfo.print_report(wfo_result)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.engine.backtest import BacktestEngine
from prop_backtest.firms.base import FirmRules
from prop_backtest.reporting.results import BacktestResult, TradeRecord
from prop_backtest.strategy.base import Strategy


@dataclass
class WFOFold:
    """Result for one walk-forward fold."""
    fold_index:       int
    best_params:      dict[str, Any]
    is_score:         float          # in-sample optimisation score (Sharpe)
    oos_result:       BacktestResult  # out-of-sample backtest result
    oos_net_pnl:      float
    oos_sharpe:       float
    oos_passed:       bool


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output."""
    folds:            list[WFOFold]
    oos_equity_curve: list[tuple]    # stitched OOS equity curve
    oos_trades:       list[TradeRecord]
    total_oos_pnl:    float
    avg_oos_sharpe:   float
    pass_rate:        float           # fraction of folds that passed
    param_stability:  dict[str, Any]  # how often each param value was selected


class WalkForwardOptimizer:
    """Rolls an in-sample/out-of-sample window across the bar series.

    Args:
        firm_rules:        Prop firm rules to enforce.
        tier_name:         Account tier.
        contract:          Futures contract spec.
        strategy_factory:  Callable(params) -> Strategy instance.
        param_grid:        List of parameter dicts to grid-search.
        is_window:         In-sample window in bars (default 120).
        oos_window:        Out-of-sample window in bars (default 30).
        step:              Step size in bars between folds (default = oos_window).
        commission_per_rt: Commission per round-turn (default $4.50).
        slippage_ticks:    Slippage in ticks (default 0).
        score_metric:      Stat key to optimise on in-sample (default "sharpe_ratio").
    """

    def __init__(
        self,
        firm_rules: FirmRules,
        tier_name: str,
        contract: ContractSpec,
        strategy_factory: Callable[[dict], Strategy],
        param_grid: list[dict[str, Any]],
        is_window: int = 120,
        oos_window: int = 30,
        step: int | None = None,
        commission_per_rt: float = 4.50,
        slippage_ticks: int = 0,
        score_metric: str = "sharpe_ratio",
    ) -> None:
        self.firm_rules       = firm_rules
        self.tier_name        = tier_name
        self.contract         = contract
        self.strategy_factory = strategy_factory
        self.param_grid       = param_grid
        self.is_window        = is_window
        self.oos_window       = oos_window
        self.step             = step if step is not None else oos_window
        self.commission_per_rt = commission_per_rt
        self.slippage_ticks   = slippage_ticks
        self.score_metric     = score_metric

    def run(self, bars: list[BarData]) -> WalkForwardResult:
        """Execute the walk-forward optimisation over all folds."""
        total = len(bars)
        min_bars = self.is_window + self.oos_window
        if total < min_bars:
            raise ValueError(
                f"Need at least {min_bars} bars for WFO "
                f"(is={self.is_window}, oos={self.oos_window}), got {total}."
            )

        folds:          list[WFOFold]   = []
        oos_equity:     list[tuple]     = []
        oos_trades:     list[TradeRecord] = []
        equity_offset   = 0.0           # stitch OOS curves end-to-end

        # Generate fold start indices
        starts = list(range(0, total - min_bars + 1, self.step))

        for fold_idx, is_start in enumerate(starts):
            is_end  = is_start + self.is_window
            oos_end = min(is_end + self.oos_window, total)

            if oos_end - is_end < 5:
                break  # not enough OOS bars

            is_bars  = bars[is_start:is_end]
            oos_bars = bars[is_end:oos_end]

            # ── Optimise on in-sample ──────────────────────────────────
            best_params, best_is_score = self._optimise(is_bars)

            # ── Evaluate best params on out-of-sample ─────────────────
            strat      = self.strategy_factory(best_params)
            oos_result = self._backtest(strat, oos_bars)
            oos_sharpe = oos_result.stats.get("sharpe_ratio", 0.0)
            oos_pnl    = oos_result.stats.get("total_net_pnl", 0.0)

            # Stitch OOS equity curve (offset so it continues from previous fold)
            tier_start = self.firm_rules.get_tier(self.tier_name).starting_balance
            if oos_equity:
                equity_offset = oos_equity[-1][1] - tier_start
            for ts, eq in oos_result.equity_curve:
                oos_equity.append((ts, eq + equity_offset))

            # Accumulate OOS trades
            for t in oos_result.trades:
                oos_trades.append(t)

            folds.append(WFOFold(
                fold_index=fold_idx,
                best_params=best_params,
                is_score=best_is_score,
                oos_result=oos_result,
                oos_net_pnl=oos_pnl,
                oos_sharpe=oos_sharpe,
                oos_passed=oos_result.passed,
            ))

        if not folds:
            raise ValueError("No folds were generated. Check is_window and oos_window.")

        # ── Aggregate ──────────────────────────────────────────────────────
        total_oos_pnl  = sum(f.oos_net_pnl  for f in folds)
        avg_oos_sharpe = sum(f.oos_sharpe    for f in folds) / len(folds)
        pass_rate      = sum(f.oos_passed    for f in folds) / len(folds)

        # Parameter stability: count how often each param value was chosen
        param_stability: dict[str, dict] = {}
        for fold in folds:
            for k, v in fold.best_params.items():
                param_stability.setdefault(k, {})
                param_stability[k][v] = param_stability[k].get(v, 0) + 1

        return WalkForwardResult(
            folds=folds,
            oos_equity_curve=oos_equity,
            oos_trades=oos_trades,
            total_oos_pnl=total_oos_pnl,
            avg_oos_sharpe=avg_oos_sharpe,
            pass_rate=pass_rate,
            param_stability=param_stability,
        )

    def _optimise(self, bars: list[BarData]) -> tuple[dict, float]:
        """Grid search param_grid on in-sample bars. Returns best params + score."""
        best_score  = float("-inf")
        best_params = self.param_grid[0]

        for params in self.param_grid:
            try:
                strat  = self.strategy_factory(params)
                result = self._backtest(strat, bars)
                score  = result.stats.get(self.score_metric, float("-inf"))
                # Penalise if drawdown was breached
                if result.stats.get("max_drawdown_dollars", 0) == 0 and not result.trades:
                    score = float("-inf")
                if score > best_score:
                    best_score  = score
                    best_params = params
            except Exception:
                continue

        return best_params, best_score

    def _backtest(self, strategy: Strategy, bars: list[BarData]) -> BacktestResult:
        engine = BacktestEngine(
            strategy=strategy,
            firm_rules=self.firm_rules,
            tier_name=self.tier_name,
            contract=self.contract,
            commission_per_rt=self.commission_per_rt,
            slippage_ticks=self.slippage_ticks,
        )
        return engine.run(bars)

    def print_report(self, wfo: WalkForwardResult) -> None:
        """Print a Rich-formatted walk-forward summary."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box
        except ImportError:
            self._print_plain(wfo)
            return

        console = Console()
        colour  = "green" if wfo.total_oos_pnl >= 0 else "red"

        console.print(Panel(
            f"[bold]Walk-Forward Optimisation[/bold]  |  "
            f"{len(wfo.folds)} folds  |  "
            f"OOS PnL: [{colour}]${wfo.total_oos_pnl:,.2f}[/{colour}]  |  "
            f"Avg Sharpe: {wfo.avg_oos_sharpe:.2f}  |  "
            f"Pass Rate: {wfo.pass_rate * 100:.0f}%",
            style="bold white",
        ))

        # Per-fold table
        t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
        t.add_column("Fold", justify="right", width=5)
        t.add_column("IS Score", justify="right", width=10)
        t.add_column("Best Params", width=28)
        t.add_column("OOS PnL", justify="right", width=12)
        t.add_column("OOS Sharpe", justify="right", width=11)
        t.add_column("Trades", justify="right", width=7)
        t.add_column("Pass", justify="center", width=5)

        for f in wfo.folds:
            pnl_str  = f"${f.oos_net_pnl:,.2f}"
            pnl_col  = "green" if f.oos_net_pnl >= 0 else "red"
            pass_str = "[green]Y[/green]" if f.oos_passed else "[red]N[/red]"
            params_short = ", ".join(f"{k}={v}" for k, v in f.best_params.items())
            n_trades = len(f.oos_result.trades)
            t.add_row(
                str(f.fold_index + 1),
                f"{f.is_score:.2f}",
                params_short,
                f"[{pnl_col}]{pnl_str}[/{pnl_col}]",
                f"{f.oos_sharpe:.2f}",
                str(n_trades),
                pass_str,
            )

        console.print(t)

        # Parameter stability
        console.print("\n[bold]Parameter Stability (selection frequency):[/bold]")
        for param, counts in wfo.param_stability.items():
            total = sum(counts.values())
            bars_str = "  ".join(
                f"{v}: {c/total*100:.0f}%" for v, c in sorted(counts.items())
            )
            console.print(f"  [cyan]{param}[/cyan]: {bars_str}")

    def _print_plain(self, wfo: WalkForwardResult) -> None:
        print(f"\n=== Walk-Forward: {len(wfo.folds)} folds ===")
        print(f"Total OOS PnL:  ${wfo.total_oos_pnl:,.2f}")
        print(f"Avg OOS Sharpe: {wfo.avg_oos_sharpe:.2f}")
        print(f"OOS Pass Rate:  {wfo.pass_rate * 100:.0f}%")
        for f in wfo.folds:
            print(f"  Fold {f.fold_index+1}: params={f.best_params} "
                  f"OOS_PnL=${f.oos_net_pnl:,.2f} pass={f.oos_passed}")
