"""Monte Carlo simulation for estimating prop firm pass rate."""

import numpy as np
from dataclasses import dataclass
from config import RULES, PARAMS


@dataclass
class MonteCarloResult:
    n_simulations: int
    pass_count: int
    fail_dd_count: int
    fail_time_count: int
    pass_rate: float
    median_days_to_pass: float
    p10_pnl: float
    p50_pnl: float
    p90_pnl: float
    avg_max_dd: float
    bust_rate: float


def simulate_evaluation(
    trade_pnls: list[float],
    trades_per_day: float = 1.5,
    max_days: int = 22,
    rng: np.random.Generator | None = None,
) -> tuple[str, float, int, float]:
    """Simulate a single prop firm evaluation by sampling trades.

    Returns: (status, final_pnl, days_used, max_drawdown)
    """
    if rng is None:
        rng = np.random.default_rng()

    balance = RULES.starting_balance
    max_eod_balance = balance
    trailing_dd_floor = balance - RULES.max_trailing_drawdown
    cumulative_pnl = 0.0
    best_day_pnl = 0.0
    max_dd = 0.0

    for day in range(1, max_days + 1):
        # Randomly determine number of trades today (1-3)
        n_trades = max(1, min(3, int(rng.poisson(trades_per_day))))
        daily_pnl = 0.0

        for _ in range(n_trades):
            # Sample a random trade P&L
            pnl = rng.choice(trade_pnls)
            daily_pnl += pnl
            cumulative_pnl += pnl
            balance += pnl

            # Track drawdown
            peak = max(max_eod_balance, balance)
            dd = peak - balance
            max_dd = max(max_dd, dd)

            # Check intraday drawdown against trailing floor
            if balance <= trailing_dd_floor:
                return "failed_drawdown", cumulative_pnl, day, max_dd

            # Daily loss gate
            if RULES.daily_loss_limit and daily_pnl <= -PARAMS.daily_loss_gate:
                break

            # Daily profit gate
            if daily_pnl >= PARAMS.daily_profit_gate:
                break

            # Check if passed
            if cumulative_pnl >= RULES.profit_target:
                # Check consistency
                best_day_pnl = max(best_day_pnl, daily_pnl)
                max_allowed = RULES.profit_target * RULES.consistency_pct
                if best_day_pnl <= max_allowed:
                    return "passed", cumulative_pnl, day, max_dd
                # If consistency violated, keep going

        # End of day: update trailing DD
        best_day_pnl = max(best_day_pnl, daily_pnl)
        if balance > max_eod_balance:
            max_eod_balance = balance
            new_floor = max_eod_balance - RULES.max_trailing_drawdown
            trailing_dd_floor = min(
                max(new_floor, trailing_dd_floor),
                RULES.starting_balance,
            )

        # Check passed at EOD
        if cumulative_pnl >= RULES.profit_target:
            max_allowed = RULES.profit_target * RULES.consistency_pct
            if best_day_pnl <= max_allowed:
                return "passed", cumulative_pnl, day, max_dd

    return "failed_time", cumulative_pnl, max_days, max_dd


def run_monte_carlo(
    trade_pnls: list[float],
    n_simulations: int = 10_000,
    trades_per_day: float = 1.5,
    max_days: int = 22,
    seed: int = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulation to estimate pass rate.

    Args:
        trade_pnls: List of individual trade P&Ls from backtesting.
        n_simulations: Number of simulated evaluations.
        trades_per_day: Average trades per day.
        max_days: Maximum trading days.
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with pass rate and statistics.
    """
    rng = np.random.default_rng(seed)

    pass_count = 0
    fail_dd_count = 0
    fail_time_count = 0
    final_pnls = []
    days_to_pass = []
    max_dds = []

    for _ in range(n_simulations):
        status, pnl, days, dd = simulate_evaluation(
            trade_pnls, trades_per_day, max_days, rng,
        )

        final_pnls.append(pnl)
        max_dds.append(dd)

        if status == "passed":
            pass_count += 1
            days_to_pass.append(days)
        elif status == "failed_drawdown":
            fail_dd_count += 1
        else:
            fail_time_count += 1

    pass_rate = pass_count / n_simulations
    median_days = float(np.median(days_to_pass)) if days_to_pass else 0

    return MonteCarloResult(
        n_simulations=n_simulations,
        pass_count=pass_count,
        fail_dd_count=fail_dd_count,
        fail_time_count=fail_time_count,
        pass_rate=pass_rate,
        median_days_to_pass=median_days,
        p10_pnl=float(np.percentile(final_pnls, 10)),
        p50_pnl=float(np.percentile(final_pnls, 50)),
        p90_pnl=float(np.percentile(final_pnls, 90)),
        avg_max_dd=float(np.mean(max_dds)),
        bust_rate=fail_dd_count / n_simulations,
    )


def run_block_bootstrap(
    daily_pnls: list[float],
    n_simulations: int = 10_000,
    max_days: int = 22,
    seed: int = 42,
) -> MonteCarloResult:
    """Block bootstrap: resample entire daily P&L blocks to preserve intraday correlation."""
    rng = np.random.default_rng(seed)

    if not daily_pnls:
        return MonteCarloResult(
            n_simulations=n_simulations, pass_count=0,
            fail_dd_count=0, fail_time_count=n_simulations,
            pass_rate=0, median_days_to_pass=0,
            p10_pnl=0, p50_pnl=0, p90_pnl=0,
            avg_max_dd=0, bust_rate=0,
        )

    pass_count = 0
    fail_dd_count = 0
    fail_time_count = 0
    final_pnls = []
    days_to_pass = []
    max_dds = []

    daily_arr = np.array(daily_pnls)

    for _ in range(n_simulations):
        # Sample max_days daily P&Ls with replacement
        sampled = rng.choice(daily_arr, size=max_days, replace=True)

        balance = RULES.starting_balance
        max_eod_balance = balance
        trailing_dd_floor = balance - RULES.max_trailing_drawdown
        cumulative = 0.0
        best_day = 0.0
        max_dd = 0.0
        status = "failed_time"
        days_used = max_days

        for day_idx, day_pnl in enumerate(sampled, 1):
            cumulative += day_pnl
            balance += day_pnl
            best_day = max(best_day, day_pnl)

            peak = max(max_eod_balance, balance)
            dd = peak - balance
            max_dd = max(max_dd, dd)

            if balance <= trailing_dd_floor:
                status = "failed_drawdown"
                days_used = day_idx
                break

            if balance > max_eod_balance:
                max_eod_balance = balance
                new_floor = max_eod_balance - RULES.max_trailing_drawdown
                trailing_dd_floor = min(
                    max(new_floor, trailing_dd_floor),
                    RULES.starting_balance,
                )

            if cumulative >= RULES.profit_target:
                max_allowed = RULES.profit_target * RULES.consistency_pct
                if best_day <= max_allowed:
                    status = "passed"
                    days_used = day_idx
                    break

        final_pnls.append(cumulative)
        max_dds.append(max_dd)

        if status == "passed":
            pass_count += 1
            days_to_pass.append(days_used)
        elif status == "failed_drawdown":
            fail_dd_count += 1
        else:
            fail_time_count += 1

    return MonteCarloResult(
        n_simulations=n_simulations,
        pass_count=pass_count,
        fail_dd_count=fail_dd_count,
        fail_time_count=fail_time_count,
        pass_rate=pass_count / n_simulations,
        median_days_to_pass=float(np.median(days_to_pass)) if days_to_pass else 0,
        p10_pnl=float(np.percentile(final_pnls, 10)),
        p50_pnl=float(np.percentile(final_pnls, 50)),
        p90_pnl=float(np.percentile(final_pnls, 90)),
        avg_max_dd=float(np.mean(max_dds)),
        bust_rate=fail_dd_count / n_simulations,
    )


def print_monte_carlo(mc: MonteCarloResult, method: str = "Trade Bootstrap"):
    """Print Monte Carlo simulation results."""
    print(f"\n{'=' * 60}")
    print(f"  MONTE CARLO SIMULATION ({method})")
    print(f"{'=' * 60}")
    print(f"  Simulations:         {mc.n_simulations:,}")
    print(f"  Pass Rate:           {mc.pass_rate * 100:.1f}%  ({mc.pass_count:,} / {mc.n_simulations:,})")
    print(f"  Bust Rate (DD):      {mc.bust_rate * 100:.1f}%  ({mc.fail_dd_count:,})")
    print(f"  Timeout Rate:        {mc.fail_time_count / mc.n_simulations * 100:.1f}%  ({mc.fail_time_count:,})")
    print()
    print(f"  Median Days to Pass: {mc.median_days_to_pass:.1f}")
    print(f"  P&L (10th pctile):   ${mc.p10_pnl:,.2f}")
    print(f"  P&L (50th pctile):   ${mc.p50_pnl:,.2f}")
    print(f"  P&L (90th pctile):   ${mc.p90_pnl:,.2f}")
    print(f"  Avg Max Drawdown:    ${mc.avg_max_dd:,.2f}")
    print(f"{'=' * 60}")
