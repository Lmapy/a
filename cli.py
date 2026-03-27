#!/usr/bin/env python3
"""Command-line interface for the prop firm backtest engine."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import click

# Ensure the repo root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent))


@click.group()
def main() -> None:
    """Prop firm challenge backtesting engine with RL support."""


# ── run command ───────────────────────────────────────────────────────────────

@main.command()
@click.option("--firm", required=True, help="Prop firm name: topstep, mff, lucid")
@click.option("--tier", required=True, help="Account tier: 50K, 100K, 150K, ...")
@click.option("--contract", required=True, help="Futures contract symbol: ES, NQ, CL, ...")
@click.option("--strategy", default=None, help=(
    "Dotted import path to Strategy class, e.g. "
    "prop_backtest.strategy.examples.sma_crossover.SMACrossover. "
    "Use 'rl' to load a trained RL model (requires --model-path)."
))
@click.option("--model-path", default=None, help="Path to a trained SB3 model (for --strategy rl).")
@click.option("--start", required=True, help="Start date, e.g. 2024-01-01")
@click.option("--end", required=True, help="End date, e.g. 2024-12-31")
@click.option("--interval", default="1d", show_default=True, help="Bar interval: 1d, 1h, 15m, ...")
@click.option("--csv", default=None, help="Path to a local CSV file instead of downloading.")
@click.option("--source", default="yfinance", show_default=True, help="Remote data source: yfinance, barchart.")
@click.option("--barchart-key", default=None, envvar="BARCHART_API_KEY", help="Barchart API key (or set BARCHART_API_KEY env var).")
@click.option("--commission", default=4.50, show_default=True, type=float, help="Commission per round-turn in $.")
@click.option("--slippage", default=0, show_default=True, type=int, help="Slippage in ticks.")
@click.option("--output", default=None, help="Directory to save trades.csv and equity_curve.csv.")
@click.option("--plot", is_flag=True, default=False, help="Show and/or save an equity curve plot.")
@click.option("--plot-path", default=None, help="File path to save plot (PNG/PDF). Implies --plot.")
def run(
    firm, tier, contract, strategy, model_path,
    start, end, interval, csv, source, barchart_key,
    commission, slippage,
    output, plot, plot_path,
) -> None:
    """Run a backtest against a prop firm challenge."""
    from prop_backtest.contracts.specs import get_contract
    from prop_backtest.data.loader import DataLoader
    from prop_backtest.engine.backtest import BacktestEngine
    from prop_backtest.firms import get_firm
    from prop_backtest.reporting.report import plot_equity_curve, print_report, save_report_csv

    # ── Resolve firm and contract ─────────────────────────────────────────
    firm_rules = get_firm(firm)
    contract_spec = get_contract(contract)

    # ── Load data ─────────────────────────────────────────────────────────
    loader = DataLoader(contract_spec)
    src_label = "local CSV" if csv else source
    click.echo(f"Loading data for {contract_spec.symbol} ({start} → {end}, {interval}) via {src_label}...")
    bars = loader.load(
        start=start, end=end, interval=interval,
        local_csv_path=csv, source=source, barchart_api_key=barchart_key,
    )
    click.echo(f"Loaded {len(bars)} bars.")

    # ── Resolve strategy ──────────────────────────────────────────────────
    strat = _load_strategy(strategy, model_path)

    # ── Run backtest ──────────────────────────────────────────────────────
    engine = BacktestEngine(
        strategy=strat,
        firm_rules=firm_rules,
        tier_name=tier,
        contract=contract_spec,
        commission_per_rt=commission,
        slippage_ticks=slippage,
    )
    click.echo(f"Running backtest ({firm_rules.firm_name} {tier})...")
    result = engine.run(bars)

    # ── Output ────────────────────────────────────────────────────────────
    print_report(result)

    if output:
        save_report_csv(result, output)

    if plot or plot_path:
        plot_equity_curve(result, output_path=plot_path, show=plot)


# ── train command ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--firm", required=True, help="Prop firm name: topstep, mff, lucid")
@click.option("--tier", required=True, help="Account tier: 50K, 100K, 150K, ...")
@click.option("--contract", required=True, help="Futures contract symbol: ES, NQ, CL, ...")
@click.option("--start", required=True, help="Start date for training data.")
@click.option("--end", required=True, help="End date for training data.")
@click.option("--interval", default="1d", show_default=True, help="Bar interval.")
@click.option("--csv", default=None, help="Local CSV data file.")
@click.option("--source", default="yfinance", show_default=True, help="Remote data source: yfinance, barchart.")
@click.option("--barchart-key", default=None, envvar="BARCHART_API_KEY", help="Barchart API key.")
@click.option("--timesteps", default=500_000, show_default=True, type=int, help="Total training steps.")
@click.option("--output", default="models/ppo_agent", show_default=True, help="Model save path (without .zip).")
@click.option("--n-envs", default=4, show_default=True, type=int, help="Parallel training environments.")
@click.option("--window", default=20, show_default=True, type=int, help="Observation window in bars.")
@click.option("--commission", default=4.50, show_default=True, type=float, help="Commission per round-turn in $.")
@click.option("--slippage", default=0, show_default=True, type=int, help="Slippage in ticks.")
def train(
    firm, tier, contract, start, end, interval, csv, source, barchart_key,
    timesteps, output, n_envs, window, commission, slippage,
) -> None:
    """Train a PPO RL agent to pass a prop firm challenge."""
    from prop_backtest.contracts.specs import get_contract
    from prop_backtest.data.loader import DataLoader
    from prop_backtest.firms import get_firm
    from prop_backtest.rl.trainer import train as _train

    firm_rules = get_firm(firm)
    contract_spec = get_contract(contract)

    loader = DataLoader(contract_spec)
    click.echo(f"Loading training data for {contract_spec.symbol} ({start} → {end})...")
    bars = loader.load(
        start=start, end=end, interval=interval,
        local_csv_path=csv, source=source, barchart_api_key=barchart_key,
    )
    click.echo(f"Loaded {len(bars)} bars. Starting training for {timesteps:,} timesteps...")

    model = _train(
        firm_rules=firm_rules,
        tier_name=tier,
        contract=contract_spec,
        bars=bars,
        total_timesteps=timesteps,
        model_path=output,
        n_envs=n_envs,
        window=window,
        commission_per_rt=commission,
        slippage_ticks=slippage,
        verbose=1,
    )

    click.echo(f"\nTraining complete. Model saved to {output}.zip")
    click.echo(
        f"\nEvaluate with:\n"
        f"  python cli.py run --firm {firm} --tier {tier} --contract {contract} "
        f"--strategy rl --model-path {output} --start <START> --end <END>"
    )


# ── list-firms / list-contracts helpers ───────────────────────────────────────

@main.command("list-firms")
def list_firms() -> None:
    """List available prop firms and their tiers."""
    from prop_backtest.firms import FIRM_REGISTRY
    seen = set()
    for key, rules in FIRM_REGISTRY.items():
        if rules.firm_name in seen:
            continue
        seen.add(rules.firm_name)
        tiers = ", ".join(t.name for t in rules.tiers)
        click.echo(f"  {rules.firm_name} (key: {key})  —  tiers: {tiers}")


@main.command("list-contracts")
def list_contracts() -> None:
    """List available futures contracts."""
    from prop_backtest.contracts.specs import CONTRACT_REGISTRY
    for sym, spec in CONTRACT_REGISTRY.items():
        click.echo(
            f"  {sym:6s}  {spec.description:30s}  "
            f"tick=${spec.tick_value:.2f}  yf={spec.yfinance_ticker}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_strategy(strategy_path: str | None, model_path: str | None):
    from prop_backtest.strategy.base import FunctionStrategy
    from prop_backtest.engine.broker import Signal

    if strategy_path is None or strategy_path.lower() == "hold":
        # Default: always hold (useful for testing data loading)
        return FunctionStrategy(lambda bar, hist, acc: Signal(action="hold"), name="HoldStrategy")

    if strategy_path.lower() == "rl":
        if not model_path:
            raise click.UsageError("--model-path is required when --strategy rl is used.")
        from prop_backtest.strategy.rl_strategy import RLStrategy
        return RLStrategy(model_path=model_path)

    # Dynamically import a strategy class
    try:
        module_path, class_name = strategy_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise click.UsageError(
            f"Could not load strategy '{strategy_path}': {e}\n"
            "Expected dotted path like: prop_backtest.strategy.examples.sma_crossover.SMACrossover"
        ) from e


if __name__ == "__main__":
    main()
