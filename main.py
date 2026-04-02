#!/usr/bin/env python3
"""Gold Price Action Strategy - Prop Firm Evaluation Backtester.

Downloads real XAUUSD tick data from Dukascopy, applies a price action /
market structure strategy, and estimates prop firm pass rate via Monte Carlo.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, RESULTS_DIR, PARAMS, RULES, CONTRACT
from data.download_data import load_data, download_github_data
from data.resample import build_multi_timeframe, save_candles, load_candles, prepare_candles, tag_sessions
from backtest.engine import run_backtest, run_walkforward_backtest
from backtest.metrics import calculate_metrics, print_metrics, print_trade_log
from backtest.monte_carlo import (
    run_monte_carlo, run_block_bootstrap, print_monte_carlo,
)


def plot_equity_curve(result, save_path: Path):
    """Plot and save equity curve."""
    if not result.equity_curve:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax1 = axes[0]
    equity = result.equity_curve
    ax1.plot(equity, linewidth=1, color="steelblue")
    ax1.axhline(y=RULES.starting_balance, color="gray", linestyle="--", alpha=0.5, label="Starting Balance")
    ax1.axhline(
        y=RULES.starting_balance + RULES.profit_target,
        color="green", linestyle="--", alpha=0.7, label=f"Target (+${RULES.profit_target:,.0f})"
    )
    ax1.axhline(
        y=RULES.starting_balance - RULES.max_trailing_drawdown,
        color="red", linestyle="--", alpha=0.7, label=f"Max DD (-${RULES.max_trailing_drawdown:,.0f})"
    )
    ax1.set_title("Equity Curve - Gold Price Action Strategy", fontsize=14)
    ax1.set_ylabel("Account Balance ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Daily P&L bars
    ax2 = axes[1]
    daily_pnls = result.tracker.daily_pnl_history
    if daily_pnls:
        colors = ["green" if p > 0 else "red" for p in daily_pnls]
        ax2.bar(range(len(daily_pnls)), daily_pnls, color=colors, alpha=0.7)
        ax2.axhline(y=0, color="gray", linewidth=0.5)
        ax2.axhline(y=RULES.profit_target * RULES.consistency_pct,
                     color="orange", linestyle="--", alpha=0.5,
                     label=f"Max Day (${RULES.profit_target * RULES.consistency_pct:,.0f})")
        ax2.set_xlabel("Trading Day")
        ax2.set_ylabel("Daily P&L ($)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Equity curve saved to {save_path}")


def plot_monte_carlo_distribution(trade_pnls, save_path: Path):
    """Plot Monte Carlo P&L distribution."""
    rng = np.random.default_rng(42)
    final_pnls = []

    for _ in range(10_000):
        n_trades = max(1, min(3, int(rng.poisson(1.5))))
        total_days = 22
        cumulative = 0.0
        for _ in range(total_days):
            day_trades = rng.choice(trade_pnls, size=n_trades, replace=True)
            cumulative += sum(day_trades)
        final_pnls.append(cumulative)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_pnls, bins=80, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(x=RULES.profit_target, color="green", linestyle="--",
               linewidth=2, label=f"Target (${RULES.profit_target:,.0f})")
    ax.axvline(x=-RULES.max_trailing_drawdown, color="red", linestyle="--",
               linewidth=2, label=f"Max DD (-${RULES.max_trailing_drawdown:,.0f})")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.5)

    pct_pass = sum(1 for p in final_pnls if p >= RULES.profit_target) / len(final_pnls) * 100
    ax.set_title(f"Monte Carlo P&L Distribution (N=10,000) - Pass Rate: {pct_pass:.1f}%", fontsize=13)
    ax.set_xlabel("Cumulative P&L ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Monte Carlo distribution saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Gold PA Strategy Backtester")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download, use cached data")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward analysis")
    parser.add_argument("--mc-sims", type=int, default=10_000,
                        help="Number of Monte Carlo simulations")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load Data ──
    print("\n" + "=" * 60)
    print("  STEP 1: DATA PIPELINE")
    print("=" * 60)

    # Load real XAUUSD data - prefer 1-min data for best entry precision
    raw_data = load_data()
    raw_data.attrs["_is_ohlcv"] = True

    # Build multi-timeframe candles from the raw data
    candle_dict = build_multi_timeframe(raw_data)

    # Select entry and HTF candles
    if "5min" in candle_dict:
        candles_entry = candle_dict["5min"]
        candles_htf = candle_dict["15min"] if "15min" in candle_dict else candle_dict["1H"]
        entry_tf = "5min"
    elif "15min" in candle_dict:
        candles_entry = candle_dict["15min"]
        candles_htf = candle_dict["1H"] if "1H" in candle_dict else candle_dict["15min"]
        entry_tf = "15min"
    else:
        candles_entry = list(candle_dict.values())[0]
        candles_htf = candles_entry
        entry_tf = list(candle_dict.keys())[0]

    print(f"[DATA] Entry candles ({entry_tf}): {len(candles_entry):,}")
    print(f"[DATA] HTF candles:              {len(candles_htf):,}")
    print(f"[DATA] Date range: {candles_entry.index.min()} to {candles_entry.index.max()}")
    print(f"[DATA] Price range: ${candles_entry['close'].min():.2f} to ${candles_entry['close'].max():.2f}")

    # ── Step 2: Run Backtest ──
    print("\n" + "=" * 60)
    print("  STEP 2: BACKTESTING")
    print("=" * 60)

    if args.walkforward:
        results = run_walkforward_backtest(candles_entry, candles_htf)
        print(f"\n[WALKFORWARD] Completed {len(results)} evaluation windows")

        pass_count = sum(1 for r in results if r.tracker.status == "passed")
        print(f"[WALKFORWARD] Pass rate: {pass_count}/{len(results)} "
              f"({pass_count/len(results)*100:.1f}%)" if results else "")

        # Use last result for detailed analysis
        if results:
            result = results[-1]
        else:
            print("[ERROR] No walkforward results produced")
            return
    else:
        result = run_backtest(candles_entry, candles_htf)

    # ── Step 3: Metrics ──
    print("\n" + "=" * 60)
    print("  STEP 3: PERFORMANCE ANALYSIS")
    print("=" * 60)

    metrics = calculate_metrics(result)
    print_metrics(metrics)

    if result.trades:
        print_trade_log(result.trades)

    # ── Step 4: Monte Carlo ──
    print("\n" + "=" * 60)
    print("  STEP 4: MONTE CARLO SIMULATION")
    print("=" * 60)

    if result.trades:
        trade_pnls = [t.pnl for t in result.trades]

        # Trade-level bootstrap
        mc_trades = run_monte_carlo(
            trade_pnls,
            n_simulations=args.mc_sims,
            trades_per_day=metrics.get("trades_per_day", 1.5),
        )
        print_monte_carlo(mc_trades, method="Trade Bootstrap")

        # Block bootstrap (daily P&L)
        daily_pnls = metrics.get("daily_pnl_history", [])
        if daily_pnls:
            mc_block = run_block_bootstrap(
                daily_pnls,
                n_simulations=args.mc_sims,
            )
            print_monte_carlo(mc_block, method="Daily Block Bootstrap")

    else:
        print("[WARNING] No trades produced - cannot run Monte Carlo")

    # ── Step 5: Plots ──
    print("\n" + "=" * 60)
    print("  STEP 5: GENERATING PLOTS")
    print("=" * 60)

    plot_equity_curve(result, RESULTS_DIR / "equity_curve.png")

    if result.trades:
        trade_pnls = [t.pnl for t in result.trades]
        plot_monte_carlo_distribution(trade_pnls, RESULTS_DIR / "monte_carlo_dist.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Strategy:        Gold Price Action (Market Structure)")
    print(f"  Contract:        {CONTRACT.name} (${CONTRACT.tick_value}/tick)")
    print(f"  Prop Firm:       {RULES.name}")
    print(f"  Data Source:     ejtraderLabs/historical-data (Real XAUUSD Data)")
    print(f"  Backtest Period: {candles_entry.index.min().date()} to {candles_entry.index.max().date()}")
    print(f"  Total Trades:    {len(result.trades)}")
    print(f"  Status:          {result.tracker.status}")
    if result.trades:
        print(f"  Trade Bootstrap Pass Rate:  {mc_trades.pass_rate * 100:.1f}%")
        if daily_pnls:
            print(f"  Block Bootstrap Pass Rate:  {mc_block.pass_rate * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
