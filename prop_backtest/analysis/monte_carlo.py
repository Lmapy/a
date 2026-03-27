"""Monte Carlo pass-probability simulator.

Bootstraps the trade sequence from a completed backtest N times and re-runs
each sequence against the prop firm rules. This answers the key question:

    "Given this strategy's trade distribution, what is the probability of
     passing the challenge — and how much of the result was luck?"

Usage:
    from prop_backtest.analysis.monte_carlo import run_monte_carlo, print_mc_report
    mc = run_monte_carlo(result, n_simulations=1000)
    print_mc_report(mc)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from prop_backtest.reporting.results import BacktestResult, TradeRecord
from prop_backtest.firms.base import FirmRules, DrawdownType


@dataclass
class MonteCarloResult:
    """Aggregated Monte Carlo simulation output."""
    n_simulations:   int
    pass_rate:       float          # fraction of sims that passed
    median_final_equity:  float
    mean_final_equity:    float
    p5_final_equity:      float     # 5th percentile (worst-case tail)
    p95_final_equity:     float     # 95th percentile (best-case tail)
    median_max_drawdown:  float
    p95_max_drawdown:     float     # 95th percentile (worst drawdown tail)
    median_trades_to_pass: Optional[float]
    # Raw per-simulation outcomes for further analysis
    final_equities: list[float] = field(default_factory=list)
    max_drawdowns:  list[float] = field(default_factory=list)
    passed_flags:   list[bool]  = field(default_factory=list)


def run_monte_carlo(
    result: BacktestResult,
    firm_rules: FirmRules,
    tier_name: str,
    n_simulations: int = 1000,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """Bootstrap the trade sequence and replay against firm rules N times.

    Each simulation:
    1. Randomly resamples (with replacement) the same number of trades.
    2. Replays them sequentially, updating equity and checking rules.
    3. Records pass/fail, final equity, and max drawdown.

    Args:
        result:        Completed BacktestResult with at least 1 trade.
        firm_rules:    The FirmRules object used in the original backtest.
        tier_name:     Tier name string, e.g. "100K".
        n_simulations: Number of bootstrap simulations (default 1000).
        seed:          Random seed for reproducibility (default 42).

    Returns:
        MonteCarloResult with aggregated statistics.
    """
    if not result.trades:
        raise ValueError("BacktestResult has no trades — cannot run Monte Carlo.")

    rng = random.Random(seed)
    tier = firm_rules.get_tier(tier_name)

    trades = result.trades
    n_trades = len(trades)
    starting = tier.starting_balance
    target   = tier.profit_target
    dd_limit = tier.max_trailing_drawdown
    daily_limit = tier.daily_loss_limit
    min_days = tier.min_trading_days

    final_equities: list[float] = []
    max_drawdowns:  list[float] = []
    passed_flags:   list[bool]  = []

    for _ in range(n_simulations):
        # Resample trades with replacement
        sim_trades = rng.choices(trades, k=n_trades)

        equity   = starting
        hwm      = starting
        max_dd   = 0.0
        blown    = False
        days_active = 0

        for trade in sim_trades:
            equity += trade.net_pnl
            if equity > hwm:
                hwm = equity
            dd = hwm - equity
            if dd > max_dd:
                max_dd = dd

            # Trailing drawdown breach
            floor = max(hwm - dd_limit, starting)  # lock-to-breakeven
            if equity < floor:
                blown = True
                break

            days_active += 1  # treat each trade as one trading day (approx)

        final_equities.append(equity)
        max_drawdowns.append(max_dd)

        # Pass conditions (simplified — no daily loss tracking in bootstrap)
        profit = equity - starting
        passed = (
            not blown
            and profit >= target
            and days_active >= min_days
        )
        passed_flags.append(passed)

    # ── Aggregate ──────────────────────────────────────────────────────────
    final_equities.sort()
    max_drawdowns.sort()
    n = len(final_equities)

    pass_rate = sum(passed_flags) / n_simulations

    # Percentile helper
    def pct(sorted_list: list[float], p: float) -> float:
        idx = max(0, min(n - 1, int(p / 100 * n)))
        return sorted_list[idx]

    trades_to_pass: list[int] = []
    for sim_idx, passed in enumerate(passed_flags):
        if passed:
            trades_to_pass.append(n_trades)  # proxy: full sequence

    return MonteCarloResult(
        n_simulations=n_simulations,
        pass_rate=pass_rate,
        median_final_equity=pct(final_equities, 50),
        mean_final_equity=sum(final_equities) / n,
        p5_final_equity=pct(final_equities, 5),
        p95_final_equity=pct(final_equities, 95),
        median_max_drawdown=pct(max_drawdowns, 50),
        p95_max_drawdown=pct(max_drawdowns, 95),
        median_trades_to_pass=float(sum(trades_to_pass) / len(trades_to_pass)) if trades_to_pass else None,
        final_equities=final_equities,
        max_drawdowns=max_drawdowns,
        passed_flags=passed_flags,
    )


def print_mc_report(mc: MonteCarloResult, starting_balance: float = 0.0) -> None:
    """Print a Rich-formatted Monte Carlo summary table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
    except ImportError:
        _print_mc_plain(mc, starting_balance)
        return

    console = Console()

    pass_pct = mc.pass_rate * 100
    colour   = "green" if pass_pct >= 60 else "yellow" if pass_pct >= 30 else "red"

    console.print(Panel(
        f"[bold]Monte Carlo Simulation[/bold]  |  "
        f"[{colour}]Pass Rate: {pass_pct:.1f}%[/{colour}]  |  "
        f"N = {mc.n_simulations:,} simulations",
        style="bold white",
    ))

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    t.add_column("Metric",          style="dim", width=36)
    t.add_column("Value",           justify="right", width=18)

    def _eq(v: float) -> str:
        return f"${v:,.2f}"

    t.add_row("Pass Rate",                    f"[{colour}]{pass_pct:.1f}%[/{colour}]")
    t.add_row("", "")
    t.add_row("Median Final Equity",          _eq(mc.median_final_equity))
    t.add_row("Mean Final Equity",            _eq(mc.mean_final_equity))
    t.add_row("5th Pct Final Equity  (bear)", _eq(mc.p5_final_equity))
    t.add_row("95th Pct Final Equity (bull)", _eq(mc.p95_final_equity))
    t.add_row("", "")
    t.add_row("Median Max Drawdown",          _eq(mc.median_max_drawdown))
    t.add_row("95th Pct Max Drawdown (tail)", _eq(mc.p95_max_drawdown))

    console.print(t)

    if pass_pct < 30:
        console.print("[red]  Strategy has <30% pass probability — edge is insufficient.[/red]")
    elif pass_pct < 60:
        console.print("[yellow]  Marginal edge — consider tighter risk management.[/yellow]")
    else:
        console.print("[green]  Strong pass probability — strategy has consistent edge.[/green]")


def _print_mc_plain(mc: MonteCarloResult, starting_balance: float) -> None:
    print(f"\n=== Monte Carlo: {mc.n_simulations} simulations ===")
    print(f"Pass Rate:             {mc.pass_rate * 100:.1f}%")
    print(f"Median Final Equity:   ${mc.median_final_equity:,.2f}")
    print(f"5th Pct Equity:        ${mc.p5_final_equity:,.2f}")
    print(f"95th Pct Equity:       ${mc.p95_final_equity:,.2f}")
    print(f"Median Max Drawdown:   ${mc.median_max_drawdown:,.2f}")
    print(f"95th Pct Max Drawdown: ${mc.p95_max_drawdown:,.2f}")
