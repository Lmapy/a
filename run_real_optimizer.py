#!/usr/bin/env python3
"""
Main runner with REAL gold data from ejtraderLabs.
Fetches real XAU/USD data, builds features, runs ML optimization loop
until a strategy passes the prop firm combine in <=5 trading days.

Data: ejtraderLabs/historical-data (GitHub) - Real OHLCV gold data 2012-2022
Prices in dataset are x100, so we divide by 100.

Anti-inflation measures:
- Real market data (not synthetic)
- Walk-forward validation (train on past, test on unseen future)
- Realistic spread ($0.30), commission ($7/lot), slippage ($0.05)
- Bar-by-bar simulation with no look-ahead bias
- Out-of-sample final validation on completely unseen data
- Multiple independent test periods for robustness
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd

from feature_engine import compute_features, prepare_ml_data
from ml_strategy import MLStrategy, StrategyOptimizer
from prop_firm_sim import PropFirmSimulator


def load_real_data(timeframe='h1'):
    """Load real gold data from CSV."""
    path = os.path.join(os.path.dirname(__file__), 'data', f'gold_{timeframe}_real.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')

    # Prices are x100 in this dataset
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] / 100.0

    # Rename to standard OHLCV format
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    })

    return df


def multi_period_validation(config, feature_cols, df, n_periods=5):
    """
    Walk-forward validation across multiple periods.
    Split data into n_periods windows, train on each and test on next.
    Strategy must be profitable in majority of periods.
    """
    period_size = len(df) // (n_periods + 1)
    results = []

    print(f"\n  Multi-period validation ({n_periods} periods, {period_size} bars each):")

    for i in range(n_periods):
        train_start = 0
        train_end = period_size * (i + 1)
        test_start = train_end
        test_end = min(test_start + period_size, len(df))

        if test_end <= test_start + 100:
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        opt = StrategyOptimizer(df, feature_cols)
        result = opt.evaluate_config(config, train_df, test_df, verbose=False)

        if result is not None:
            status = "PASS" if result.passed else f"FAIL({result.reason[:15]})"
            print(f"    Period {i+1}: {status} | "
                  f"P&L: {result.total_pnl_pct:+.2f}% | "
                  f"Days: {result.days_taken} | "
                  f"WR: {result.win_rate:.0f}% | "
                  f"PF: {result.profit_factor:.2f} | "
                  f"MaxDD: ${result.max_total_dd:,.0f}")
            results.append(result)

    passed = sum(1 for r in results if r.passed)
    profitable = sum(1 for r in results if r.total_pnl > 0)

    print(f"    Summary: {passed}/{len(results)} passed, {profitable}/{len(results)} profitable")
    return results, passed, profitable


def main():
    start_time = time.time()

    print("=" * 70)
    print("GOLD XAU/USD ML TRADING STRATEGY OPTIMIZER")
    print("Using REAL market data from ejtraderLabs (2012-2022)")
    print("Target: Pass prop firm combine in <=5 trading days")
    print("=" * 70)

    # Load all available real data
    datasets = {}
    for tf in ['m15', 'm30', 'h1', 'h4', 'd1']:
        try:
            df = load_real_data(tf)
            datasets[tf] = df
            print(f"  {tf}: {len(df):,} bars, ${df['Close'].iloc[0]:.2f} to ${df['Close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"  {tf}: Failed to load: {e}")

    # Use m15 as primary (highest resolution with sufficient history)
    # m15 has 230k bars over 10 years - excellent for ML
    primary_tf = 'm15'
    if primary_tf not in datasets:
        primary_tf = 'h1'

    primary_df = datasets[primary_tf]
    print(f"\nPrimary dataset: {primary_tf} ({len(primary_df):,} bars)")
    print(f"Date range: {primary_df.index[0]} to {primary_df.index[-1]}")

    # Prepare features
    print("\nComputing features...")
    df_feat, feature_cols = prepare_ml_data(primary_df, target_bars_ahead=10, min_move_pct=0.1)
    print(f"Features: {len(feature_cols)}, Samples: {len(df_feat):,}")

    # Run optimization
    print(f"\n{'='*70}")
    print("ML OPTIMIZATION LOOP")
    print("Testing model types: XGBoost, LightGBM, RandomForest")
    print("Walk-forward validation with out-of-sample testing")
    print(f"{'='*70}")

    optimizer = StrategyOptimizer(df_feat, feature_cols)
    best_overall = None
    best_score = -9999
    passing_strategies = []

    for round_num in range(10):  # Up to 10 rounds
        print(f"\n{'#'*70}")
        print(f"ROUND {round_num + 1}/10")
        print(f"{'#'*70}")

        result = optimizer.optimize(max_iterations=20, verbose=True)

        if result['status'] == 'PASSED':
            config = result['best_config']
            print(f"\n>>> FOUND PASSING STRATEGY ON PRIMARY DATA!")

            # Multi-period robustness check on primary timeframe
            mp_results, mp_passed, mp_profitable = multi_period_validation(
                config, feature_cols, df_feat, n_periods=5
            )

            # Cross-timeframe validation
            print(f"\n  Cross-timeframe validation:")
            tf_passed = 0
            tf_total = 0
            for tf_name, tf_df in datasets.items():
                if tf_name == primary_tf or len(tf_df) < 500:
                    continue
                tf_total += 1

                tf_feat = compute_features(tf_df).dropna()
                if len(tf_feat) < 300:
                    continue

                split = int(len(tf_feat) * 0.6)
                opt2 = StrategyOptimizer(tf_feat, feature_cols)
                tf_result = opt2.evaluate_config(config, tf_feat.iloc[:split], tf_feat.iloc[split:], verbose=False)

                if tf_result:
                    status = "PASS" if tf_result.passed else f"FAIL"
                    print(f"    {tf_name}: {status} | P&L: {tf_result.total_pnl_pct:+.2f}% | "
                          f"Days: {tf_result.days_taken} | WR: {tf_result.win_rate:.0f}% | PF: {tf_result.profit_factor:.2f}")
                    if tf_result.passed or tf_result.total_pnl > 0:
                        tf_passed += 1

            is_robust = mp_profitable >= 3 and tf_passed >= 1

            passing_strategies.append({
                'config': config,
                'result': result,
                'mp_passed': mp_passed,
                'mp_profitable': mp_profitable,
                'tf_passed': tf_passed,
                'robust': is_robust,
            })

            if is_robust:
                print(f"\n{'*'*70}")
                print(f"ROBUST STRATEGY FOUND!")
                print(f"Multi-period: {mp_passed}/5 passed, {mp_profitable}/5 profitable")
                print(f"Cross-timeframe: {tf_passed}/{tf_total} profitable")
                print(f"{'*'*70}")
                best_overall = result
                break

        # Track best effort
        if result.get('best_config'):
            score = optimizer.score_result(result.get('test_result') or result.get('val_result'))
            if score > best_score:
                best_score = score
                best_overall = result

    # Print final report
    if passing_strategies:
        # Pick the most robust one
        passing_strategies.sort(key=lambda x: (x['robust'], x['mp_profitable'], x['tf_passed']), reverse=True)
        best = passing_strategies[0]
        config = best['config']
        result = best['result']
    elif best_overall and best_overall.get('best_config'):
        config = best_overall['best_config']
        result = best_overall
    else:
        print("\nNo viable strategy found after all rounds.")
        return

    # Final report
    print(f"\n{'='*70}")
    print("FINAL STRATEGY REPORT")
    print(f"{'='*70}")

    print(f"\n--- Strategy Configuration ---")
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
        print(f"  Trading days: {res.days_taken}")
        print(f"  Final balance: ${res.final_balance:,.2f}")
        print(f"  Total P&L: ${res.total_pnl:,.2f} ({res.total_pnl_pct:.2f}%)")
        print(f"  Max daily DD: ${res.max_daily_dd:,.2f} ({res.max_daily_dd/1000:.1f}%)")
        print(f"  Max total DD: ${res.max_total_dd:,.2f} ({res.max_total_dd/1000:.1f}%)")
        print(f"  Total trades: {res.total_trades}")
        print(f"  Win rate: {res.win_rate:.1f}%")
        print(f"  Avg win: ${res.avg_win:,.2f}")
        print(f"  Avg loss: ${res.avg_loss:,.2f}")
        print(f"  Profit factor: {res.profit_factor:.2f}")

        if res.daily_results:
            print(f"\n  --- Daily Breakdown ---")
            for day in res.daily_results[:10]:
                status = "+" if day.pnl >= 0 else "-"
                print(f"    {day.date}: {status}${abs(day.pnl):,.2f} ({day.pnl_pct:+.2f}%) | "
                      f"Trades: {day.num_trades} W:{day.winning_trades} L:{day.losing_trades}")
            if len(res.daily_results) > 10:
                print(f"    ... ({len(res.daily_results) - 10} more days)")

    # Save results
    output = {
        'config': {k: v for k, v in config.items() if k != 'model_params'},
        'status': result['status'],
        'data_source': 'ejtraderLabs/historical-data (REAL XAU/USD data 2012-2022)',
        'primary_timeframe': primary_tf,
        'anti_inflation_measures': [
            'Real market data (not synthetic)',
            'Walk-forward validation',
            'Out-of-sample testing on unseen data',
            'Multi-period robustness check (5 periods)',
            'Cross-timeframe validation',
            'Realistic costs: $0.30 spread, $7/lot commission, $0.05 slippage',
            'Bar-by-bar simulation with no look-ahead bias',
            'Dynamic position sizing with ATR-based stops',
        ],
    }

    for key in ['val_result', 'test_result', 'final_oos_result']:
        res = result.get(key)
        if res:
            output[key] = {
                'passed': res.passed,
                'reason': res.reason,
                'days_taken': res.days_taken,
                'final_balance': round(res.final_balance, 2),
                'total_pnl': round(res.total_pnl, 2),
                'total_pnl_pct': round(res.total_pnl_pct, 2),
                'max_daily_dd': round(res.max_daily_dd, 2),
                'max_total_dd': round(res.max_total_dd, 2),
                'total_trades': res.total_trades,
                'win_rate': round(res.win_rate, 1),
                'avg_win': round(res.avg_win, 2),
                'avg_loss': round(res.avg_loss, 2),
                'profit_factor': round(res.profit_factor, 2),
            }

    path = os.path.join(os.path.dirname(__file__), 'strategy_results.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")

    elapsed = time.time() - start_time
    print(f"\nTotal optimization time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
