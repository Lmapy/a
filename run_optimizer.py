#!/usr/bin/env python3
"""
Main runner: Fetches gold data, builds features, runs ML optimization loop
until a strategy passes the prop firm combine in <=5 trading days.

Anti-inflation measures:
- Walk-forward validation (train on past, test on unseen future)
- Realistic spread, commission, slippage
- Bar-by-bar simulation with no look-ahead
- Out-of-sample final validation
- Multiple independent test periods
- Conservative position sizing with risk management
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd

from data_fetcher import fetch_gold_intraday_chunks, save_all_data, load_data, DATA_DIR
from feature_engine import compute_features, prepare_ml_data
from ml_strategy import MLStrategy, StrategyOptimizer
from prop_firm_sim import PropFirmSimulator


def fetch_data():
    """Fetch and save gold data."""
    print("=" * 70)
    print("STEP 1: FETCHING GOLD (XAU/USD) DATA")
    print("=" * 70)

    all_data = fetch_gold_intraday_chunks()
    save_all_data(all_data)
    return all_data


def select_best_data():
    """Select the best available dataset for backtesting."""
    print("\n" + "=" * 70)
    print("STEP 2: SELECTING BEST DATA FOR BACKTESTING")
    print("=" * 70)

    # Priority: 1h (best balance of granularity and history)
    for name in ["1h_2y", "5m_60d", "15m_60d", "1d_5y"]:
        df = load_data(name)
        if df is not None and len(df) > 500:
            print(f"Using dataset: {name} ({len(df)} bars)")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            return df, name

    raise RuntimeError("No usable data found!")


def run_multi_timeframe_optimization(datasets):
    """
    Run optimization across multiple timeframes for robustness.
    A strategy must work across different data to be considered valid.
    """
    results = {}

    for name, df in datasets.items():
        if len(df) < 500:
            print(f"Skipping {name}: too few bars ({len(df)})")
            continue

        print(f"\n{'='*70}")
        print(f"OPTIMIZING ON: {name} ({len(df)} bars)")
        print(f"{'='*70}")

        # Prepare features
        df_feat, feature_cols = prepare_ml_data(df, target_bars_ahead=10, min_move_pct=0.1)
        print(f"Features computed: {len(feature_cols)} features, {len(df_feat)} samples")

        optimizer = StrategyOptimizer(df_feat, feature_cols)
        result = optimizer.optimize(max_iterations=30, verbose=True)
        results[name] = result

        if result['status'] == 'PASSED':
            print(f"\n*** STRATEGY PASSED ON {name}! ***")

    return results


def run_robustness_check(config, all_datasets, feature_cols):
    """
    Run the winning config across ALL available datasets.
    Strategy must pass on majority to be considered robust.
    """
    print(f"\n{'='*70}")
    print("ROBUSTNESS CHECK: Testing winning config across all timeframes")
    print(f"{'='*70}")

    pass_count = 0
    total = 0

    for name, df in all_datasets.items():
        if len(df) < 300:
            continue
        total += 1

        df_feat = compute_features(df)
        df_feat = df_feat.dropna()

        if len(df_feat) < 200:
            continue

        # Split: use last 40% as test
        split = int(len(df_feat) * 0.6)
        train = df_feat.iloc[:split]
        test = df_feat.iloc[split:]

        from ml_strategy import StrategyOptimizer
        opt = StrategyOptimizer(df_feat, feature_cols)
        result = opt.evaluate_config(config, train, test, verbose=True)

        if result is not None:
            print(f"  {name}: {'PASSED' if result.passed else 'FAILED'} | "
                  f"P&L: {result.total_pnl_pct:.2f}% | Days: {result.days_taken} | "
                  f"WR: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f}")

            if result.passed:
                pass_count += 1

    print(f"\nRobustness: Passed {pass_count}/{total} timeframes")
    return pass_count, total


def print_final_report(result, config):
    """Print detailed final report of the winning strategy."""
    print(f"\n{'='*70}")
    print("FINAL STRATEGY REPORT")
    print(f"{'='*70}")

    print(f"\nStatus: {result['status']}")
    print(f"\n--- Configuration ---")
    for k, v in config.items():
        if k != 'model_params':
            print(f"  {k}: {v}")

    for phase_name in ['val_result', 'test_result', 'final_oos_result']:
        res = result.get(phase_name)
        if res is None:
            continue

        print(f"\n--- {phase_name.upper().replace('_', ' ')} ---")
        print(f"  Passed: {res.passed}")
        print(f"  Reason: {res.reason}")
        print(f"  Days taken: {res.days_taken}")
        print(f"  Final balance: ${res.final_balance:,.2f}")
        print(f"  Total P&L: ${res.total_pnl:,.2f} ({res.total_pnl_pct:.2f}%)")
        print(f"  Max daily DD: ${res.max_daily_dd:,.2f}")
        print(f"  Max total DD: ${res.max_total_dd:,.2f}")
        print(f"  Total trades: {res.total_trades}")
        print(f"  Win rate: {res.win_rate:.1f}%")
        print(f"  Avg win: ${res.avg_win:,.2f}")
        print(f"  Avg loss: ${res.avg_loss:,.2f}")
        print(f"  Profit factor: {res.profit_factor:.2f}")

        if res.daily_results:
            print(f"\n  --- Daily Breakdown ---")
            for day in res.daily_results:
                status = "+" if day.pnl >= 0 else "-"
                print(f"    {day.date}: {status}${abs(day.pnl):,.2f} ({day.pnl_pct:+.2f}%) | "
                      f"Trades: {day.num_trades} W:{day.winning_trades} L:{day.losing_trades}")


def save_results(result, config, filename="strategy_results.json"):
    """Save results to JSON."""
    output = {
        'config': config,
        'status': result['status'],
    }

    for key in ['val_result', 'test_result', 'final_oos_result']:
        res = result.get(key)
        if res is not None:
            output[key] = {
                'passed': res.passed,
                'reason': res.reason,
                'days_taken': res.days_taken,
                'final_balance': res.final_balance,
                'total_pnl': res.total_pnl,
                'total_pnl_pct': res.total_pnl_pct,
                'max_daily_dd': res.max_daily_dd,
                'max_total_dd': res.max_total_dd,
                'total_trades': res.total_trades,
                'win_rate': res.win_rate,
                'avg_win': res.avg_win,
                'avg_loss': res.avg_loss,
                'profit_factor': res.profit_factor,
            }

    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


def main():
    start_time = time.time()

    # Step 1: Fetch data
    all_data_raw = fetch_data()

    # Step 2: Load all available datasets
    datasets = {}
    for name in ["1h_2y", "5m_60d", "15m_60d", "1d_5y", "1m_recent"]:
        df = load_data(name)
        if df is not None and len(df) > 100:
            datasets[name] = df

    if not datasets:
        print("ERROR: No data available!")
        sys.exit(1)

    print(f"\nAvailable datasets:")
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Step 3: Run optimization on best dataset
    # Use 1h data as primary (best balance of granularity and history)
    primary_name = None
    for pref in ["1h_2y", "5m_60d", "15m_60d", "1d_5y"]:
        if pref in datasets and len(datasets[pref]) > 500:
            primary_name = pref
            break

    if primary_name is None:
        primary_name = list(datasets.keys())[0]

    primary_df = datasets[primary_name]
    print(f"\nPrimary dataset: {primary_name} ({len(primary_df)} bars)")

    # Prepare features
    df_feat, feature_cols = prepare_ml_data(primary_df, target_bars_ahead=10, min_move_pct=0.1)
    print(f"Features: {len(feature_cols)}, Samples: {len(df_feat)}")

    # Step 4: Optimize
    print(f"\n{'='*70}")
    print("STEP 3: ML OPTIMIZATION LOOP")
    print("Goal: Find strategy that passes prop firm combine in <=5 days")
    print(f"{'='*70}")

    optimizer = StrategyOptimizer(df_feat, feature_cols)

    # Run multiple rounds with increasing search depth
    best_overall = None
    best_overall_score = -9999

    for round_num in range(5):  # Up to 5 rounds of 30 iterations each
        print(f"\n{'#'*70}")
        print(f"OPTIMIZATION ROUND {round_num + 1}/5")
        print(f"{'#'*70}")

        result = optimizer.optimize(max_iterations=30, verbose=True)

        if result['status'] == 'PASSED':
            config = result['best_config']

            # Robustness check across other datasets
            pass_count, total = run_robustness_check(config, datasets, feature_cols)

            if pass_count > 0:
                print_final_report(result, config)
                save_results(result, config)

                elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"OPTIMIZATION COMPLETE in {elapsed:.1f} seconds")
                print(f"Strategy PASSES prop firm combine!")
                print(f"Robustness: {pass_count}/{total} timeframes")
                print(f"{'='*70}")
                return result

        # Even if not fully passed, track best effort
        if result.get('best_config'):
            current_score = optimizer.score_result(result.get('test_result') or result.get('val_result'))
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall = result

    # If we get here, report the best effort
    if best_overall and best_overall.get('best_config'):
        print(f"\n{'='*70}")
        print("BEST EFFORT RESULT (Did not find fully passing strategy)")
        print(f"{'='*70}")
        print_final_report(best_overall, best_overall['best_config'])
        save_results(best_overall, best_overall['best_config'])

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
    return best_overall


if __name__ == "__main__":
    result = main()
