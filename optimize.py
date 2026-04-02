#!/usr/bin/env python3
"""Parameter optimizer: grid search over key strategy parameters."""

import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from data.download_data import load_data
from data.resample import build_multi_timeframe, tag_sessions
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics
from backtest.monte_carlo import run_monte_carlo
from risk.prop_firm_rules import EvaluationStatus
import config


def run_optimization():
    """Grid search over key parameters to maximize pass rate."""

    # Load data once
    print("Loading data...")
    raw = load_data()
    raw.attrs["_is_ohlcv"] = True
    candle_dict = build_multi_timeframe(raw)
    candles_entry = candle_dict["5min"]
    candles_htf = candle_dict["15min"]
    print(f"Data loaded: {len(candles_entry):,} entry candles\n")

    # Parameter grid
    param_grid = {
        "risk_per_trade": [200, 300, 400],
        "reward_risk_ratio": [1.5, 2.0, 2.5],
        "min_confluence_score": [3, 4],
        "swing_lookback": [3, 5],
        "daily_loss_gate": [500, 700],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"Testing {len(combos)} parameter combinations...")
    print(f"{'#':>3} {'Risk':>6} {'R:R':>5} {'Conf':>5} {'Swing':>6} {'DLoss':>6} | "
          f"{'Trades':>7} {'WR%':>5} {'PF':>5} {'P&L':>8} {'Status':>10} {'MC%':>5}")
    print("-" * 95)

    results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Apply parameters
        config.PARAMS.risk_per_trade = params["risk_per_trade"]
        config.PARAMS.reward_risk_ratio = params["reward_risk_ratio"]
        config.PARAMS.min_confluence_score = params["min_confluence_score"]
        config.PARAMS.swing_lookback = params["swing_lookback"]
        config.PARAMS.daily_loss_gate = params["daily_loss_gate"]

        try:
            result = run_backtest(candles_entry, candles_htf, max_days=22)
            metrics = calculate_metrics(result)

            # Quick Monte Carlo (1000 sims for speed)
            mc_rate = 0.0
            if result.trades:
                trade_pnls = [t.pnl for t in result.trades]
                mc = run_monte_carlo(trade_pnls, n_simulations=1000,
                                     trades_per_day=metrics.get("trades_per_day", 1.5))
                mc_rate = mc.pass_rate

            wr = metrics.get("win_rate", 0) * 100
            pf = metrics.get("profit_factor", 0)
            pnl = metrics.get("cumulative_pnl", 0)
            n_trades = metrics.get("total_trades", 0)
            status = metrics.get("status", "?")

            print(f"{idx+1:>3} {params['risk_per_trade']:>6.0f} {params['reward_risk_ratio']:>5.1f} "
                  f"{params['min_confluence_score']:>5} {params['swing_lookback']:>6} "
                  f"{params['daily_loss_gate']:>6.0f} | "
                  f"{n_trades:>7} {wr:>5.1f} {pf:>5.2f} ${pnl:>7.0f} {status:>10} {mc_rate*100:>5.1f}")

            results.append({
                **params,
                "trades": n_trades,
                "win_rate": wr,
                "profit_factor": pf,
                "pnl": pnl,
                "status": status,
                "mc_pass_rate": mc_rate,
                "max_dd": metrics.get("max_drawdown", 0),
            })

        except Exception as e:
            print(f"{idx+1:>3} ERROR: {e}")

    # Sort by MC pass rate
    results.sort(key=lambda x: x["mc_pass_rate"], reverse=True)

    print("\n" + "=" * 70)
    print("  TOP 5 CONFIGURATIONS BY MC PASS RATE")
    print("=" * 70)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: MC Pass Rate = {r['mc_pass_rate']*100:.1f}%")
        print(f"    Risk/Trade: ${r['risk_per_trade']:.0f}")
        print(f"    R:R Ratio:  {r['reward_risk_ratio']:.1f}")
        print(f"    Confluence: {r['min_confluence_score']}")
        print(f"    Swing LB:   {r['swing_lookback']}")
        print(f"    Daily Loss: ${r['daily_loss_gate']:.0f}")
        print(f"    Win Rate:   {r['win_rate']:.1f}%")
        print(f"    PF:         {r['profit_factor']:.2f}")
        print(f"    Trades:     {r['trades']}")
        print(f"    P&L:        ${r['pnl']:.0f}")
        print(f"    Max DD:     ${r['max_dd']:.0f}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/optimization_results.csv", index=False)
    print(f"\nResults saved to results/optimization_results.csv")

    return results


if __name__ == "__main__":
    run_optimization()
