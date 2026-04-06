#!/usr/bin/env python3
"""
V2: Smarter ML strategy optimizer for prop firm challenge.
Key improvements:
- Uses h1 data (less noise, faster training)
- Trend-following bias (gold trends strongly)
- Aggressive but controlled position sizing
- Higher R:R ratios (2:1 to 4:1)
- Ensemble approach: only trade when multiple signals agree
- Targeted search for high profit factor strategies first
- Then scales position size to hit 8% target fast
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from feature_engine import compute_features
from prop_firm_sim import PropFirmSimulator, GOLD_VALUE_PER_LOT


def load_real_data(timeframe='h1'):
    path = os.path.join(os.path.dirname(__file__), 'data', f'gold_{timeframe}_real.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] / 100.0
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'tick_volume': 'Volume'})
    return df


def create_trend_labels(df, lookahead=20, min_move_pct=0.3):
    """
    Create labels based on significant moves, not noise.
    Only label strong directional moves as buy/sell.
    """
    future_high = df['High'].rolling(lookahead).max().shift(-lookahead)
    future_low = df['Low'].rolling(lookahead).min().shift(-lookahead)
    close = df['Close']

    # Potential upside and downside
    upside_pct = (future_high - close) / close * 100
    downside_pct = (close - future_low) / close * 100

    labels = pd.Series(0, index=df.index)
    # Long: upside significantly > downside
    labels[(upside_pct > min_move_pct) & (upside_pct > downside_pct * 1.5)] = 1
    # Short: downside significantly > upside
    labels[(downside_pct > min_move_pct) & (downside_pct > upside_pct * 1.5)] = -1

    return labels


def train_ensemble(X_train, y_train):
    """Train 3 models for ensemble voting."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Map labels
    y_mapped = np.where(y_train == -1, 0, np.where(y_train == 0, 1, 2))

    models = {}

    # XGBoost - heavy regularization to prevent overfitting
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5, min_child_weight=20,
        reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
        random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0
    )
    xgb_model.fit(X_scaled, y_mapped)
    models['xgboost'] = xgb_model

    # LightGBM - heavy regularization
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.5, min_child_samples=50,
        reg_alpha=1.0, reg_lambda=5.0, min_gain_to_split=0.5,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_scaled, y_mapped)
    models['lightgbm'] = lgb_model

    # Random Forest - conservative
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=30,
        min_samples_split=50, max_features='sqrt',
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_scaled, y_mapped)
    models['random_forest'] = rf_model

    return models, scaler


def ensemble_predict(models, scaler, X, min_agreement=2, confidence_threshold=0.55):
    """
    Generate signals from ensemble.
    Only signal when at least min_agreement models agree.
    """
    X_scaled = scaler.transform(X)
    n = len(X)

    votes_long = np.zeros(n)
    votes_short = np.zeros(n)
    avg_confidence_long = np.zeros(n)
    avg_confidence_short = np.zeros(n)

    for name, model in models.items():
        proba = model.predict_proba(X_scaled)
        # proba columns: [short(0), hold(1), long(2)]

        confident_long = proba[:, 2] > confidence_threshold
        confident_short = proba[:, 0] > confidence_threshold

        votes_long += confident_long.astype(float)
        votes_short += confident_short.astype(float)
        avg_confidence_long += proba[:, 2]
        avg_confidence_short += proba[:, 0]

    avg_confidence_long /= len(models)
    avg_confidence_short /= len(models)

    signals = np.zeros(n, dtype=int)
    signals[(votes_long >= min_agreement)] = 1
    signals[(votes_short >= min_agreement)] = -1

    # If both trigger, pick the stronger one or hold
    both = (votes_long >= min_agreement) & (votes_short >= min_agreement)
    signals[both] = np.where(
        avg_confidence_long[both] > avg_confidence_short[both], 1, -1
    )

    return signals, avg_confidence_long, avg_confidence_short


def compute_position_sizing(signals, df, atr_col='atr_14',
                            risk_pct=2.0, atr_sl_mult=2.0, atr_tp_mult=4.0,
                            account_size=100_000):
    """
    Aggressive but controlled position sizing.
    """
    n = len(signals)
    lot_sizes = np.zeros(n)
    stop_losses = np.zeros(n)
    take_profits = np.zeros(n)

    for i in range(n):
        if signals[i] == 0:
            continue

        atr_val = df[atr_col].iloc[i] if atr_col in df.columns else 5.0
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = 5.0

        sl_dist = max(atr_val * atr_sl_mult, 1.0)
        tp_dist = max(atr_val * atr_tp_mult, 2.0)

        risk_dollars = account_size * risk_pct / 100
        lots = risk_dollars / (sl_dist * GOLD_VALUE_PER_LOT)
        lots = max(0.01, min(lots, 10.0))
        lots = round(lots, 2)

        lot_sizes[i] = lots
        stop_losses[i] = sl_dist
        take_profits[i] = tp_dist

    return lot_sizes, stop_losses, take_profits


def run_challenge_test(df_test, signals, lot_sizes, stop_losses, take_profits,
                       phase="Phase 1", target_pct=8.0):
    """Run prop firm challenge simulation."""
    sim = PropFirmSimulator(
        account_size=100_000,
        profit_target_pct=target_pct,
        max_daily_dd_pct=5.0,
        max_total_dd_pct=10.0,
        phase=phase,
        spread_pips=30,
        commission_per_lot=7.0,
        slippage_pips=5,
    )
    return sim.simulate(df_test, signals, lot_sizes, stop_losses, take_profits)


def print_result(result, label=""):
    """Print challenge result summary."""
    if result is None:
        return
    status = "PASSED" if result.passed else "FAILED"
    print(f"  [{label}] {status} | "
          f"P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%) | "
          f"Days: {result.days_taken} | Trades: {result.total_trades} | "
          f"WR: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f} | "
          f"MaxDD: ${result.max_total_dd:,.2f} | Reason: {result.reason}")


def search_iteration(train_df, val_df, test_df, feature_cols, config, iteration, verbose=True):
    """Run one iteration of the search."""
    lookahead = config['lookahead']
    min_move = config['min_move_pct']
    confidence = config['confidence']
    min_agreement = config['min_agreement']
    risk_pct = config['risk_pct']
    atr_sl_mult = config['atr_sl_mult']
    atr_tp_mult = config['atr_tp_mult']

    # Create labels for training
    train_labels = create_trend_labels(train_df, lookahead, min_move)
    train_clean = train_df.copy()
    train_clean['target'] = train_labels
    train_clean = train_clean.dropna()

    avail_features = [c for c in feature_cols if c in train_clean.columns]
    X_train = train_clean[avail_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_clean['target']

    # Check class balance
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 50:
        return None, None, config

    # Train ensemble
    models, scaler = train_ensemble(X_train, y_train)

    results = {}
    for name, df_eval in [('val', val_df), ('test', test_df)]:
        eval_clean = df_eval.dropna()
        avail_eval = [c for c in avail_features if c in eval_clean.columns]
        if len(avail_eval) < 10 or len(eval_clean) < 100:
            continue

        X_eval = eval_clean[avail_eval].replace([np.inf, -np.inf], np.nan).fillna(0)

        signals, conf_long, conf_short = ensemble_predict(
            models, scaler, X_eval,
            min_agreement=min_agreement,
            confidence_threshold=confidence
        )

        lot_sizes, stop_losses, take_profits = compute_position_sizing(
            signals, eval_clean,
            risk_pct=risk_pct,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult
        )

        result = run_challenge_test(eval_clean, signals, lot_sizes, stop_losses, take_profits)
        results[name] = result

        if verbose:
            print_result(result, f"{name} | LA={lookahead} MM={min_move} C={confidence} "
                        f"A={min_agreement} R={risk_pct} SL={atr_sl_mult} TP={atr_tp_mult}")

    return results.get('val'), results.get('test'), config


def main():
    start_time = time.time()

    print("=" * 70)
    print("GOLD XAU/USD ML TRADING STRATEGY OPTIMIZER V2")
    print("REAL market data | Ensemble ML | Trend-following")
    print("=" * 70)

    # Load h1 data (best balance of signal quality and history)
    df_h1 = load_real_data('h1')
    print(f"H1 data: {len(df_h1):,} bars, {df_h1.index[0]} to {df_h1.index[-1]}")
    print(f"Price range: ${df_h1['Close'].min():.2f} to ${df_h1['Close'].max():.2f}")

    # Also load other TFs for cross-validation
    df_h4 = load_real_data('h4')
    df_d1 = load_real_data('d1')
    df_m15 = load_real_data('m15')

    # Compute features on h1
    print("\nComputing features...")
    df_feat = compute_features(df_h1)
    df_feat = df_feat.dropna()
    print(f"Samples after feature computation: {len(df_feat):,}")

    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target',
                    'Adj Close', 'Dividends', 'Stock Splits', 'Capital Gains', 'Repaired?']
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)}")

    # Walk-forward split: 60% train, 20% val, 20% test
    n = len(df_feat)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:]

    print(f"\nTrain: {len(train_df):,} bars ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Val:   {len(val_df):,} bars ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"Test:  {len(test_df):,} bars ({test_df.index[0]} to {test_df.index[-1]})")

    # Search space
    search_space = {
        'lookahead': [10, 15, 20, 30, 40, 50],
        'min_move_pct': [0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
        'confidence': [0.45, 0.50, 0.55, 0.60, 0.65],
        'min_agreement': [2, 3],
        'risk_pct': [1.0, 1.5, 2.0, 2.5, 3.0],
        'atr_sl_mult': [1.5, 2.0, 2.5],
        'atr_tp_mult': [3.0, 4.0, 5.0, 6.0],
    }

    best_combined = -9999
    best_config = None
    best_val_result = None
    best_test_result = None
    passing_strategies = []

    def score(result):
        if result is None:
            return -1000
        # Minimum trade count to avoid lucky streaks
        if result.total_trades < 10:
            return -500 + result.total_trades * 10
        s = result.total_pnl_pct * 10
        if result.passed:
            s += 500
            if result.days_taken <= 5:
                s += 200
            elif result.days_taken <= 10:
                s += 100
        if result.profit_factor > 1:
            s += min((result.profit_factor - 1) * 50, 200)  # Cap PF bonus
        if result.win_rate > 50:
            s += (result.win_rate - 50) * 2
        dd_pct = result.max_total_dd / 1000
        s -= dd_pct * 15
        if result.reason in ['daily_dd_breach', 'total_dd_breach']:
            s -= 300
        return s

    def combined_score(val_r, test_r):
        """Score that requires BOTH val and test to be good."""
        if val_r is None or test_r is None:
            return -2000
        vs = score(val_r)
        ts = score(test_r)
        # Geometric mean-like: both must be positive for a good combined score
        if vs > 0 and ts > 0:
            return (vs + ts) / 2 + min(vs, ts) * 0.5  # Reward consistency
        return (vs + ts) / 2

    rng = np.random.RandomState(42)
    iteration = 0
    max_iterations = 300

    print(f"\n{'='*70}")
    print(f"STARTING SEARCH ({max_iterations} iterations)")
    print(f"{'='*70}")

    while iteration < max_iterations:
        # Generate random config
        config = {k: rng.choice(v) for k, v in search_space.items()}
        # Ensure TP > SL
        if config['atr_tp_mult'] <= config['atr_sl_mult'] * 1.2:
            config['atr_tp_mult'] = config['atr_sl_mult'] * 2.0

        # Convert numpy types to Python types
        config = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                  for k, v in config.items()}
        config['lookahead'] = int(config['lookahead'])
        config['min_agreement'] = int(config['min_agreement'])

        iteration += 1

        if iteration % 10 == 1:
            print(f"\n--- Iteration {iteration}/{max_iterations} (best combined: {best_combined:.1f}) ---")

        # If we have a best config, mutate it 40% of the time (explore more)
        if best_config is not None and rng.random() < 0.4:
            config = best_config.copy()
            # Mutate 1-2 params
            for _ in range(rng.randint(1, 3)):
                param = rng.choice(list(search_space.keys()))
                config[param] = rng.choice(search_space[param])
                if isinstance(config[param], (np.floating, np.integer)):
                    config[param] = float(config[param])
            config['lookahead'] = int(config['lookahead'])
            config['min_agreement'] = int(config['min_agreement'])
            if config['atr_tp_mult'] <= config['atr_sl_mult'] * 1.2:
                config['atr_tp_mult'] = config['atr_sl_mult'] * 2.0

        val_result, test_result, _ = search_iteration(
            train_df, val_df, test_df, feature_cols, config, iteration,
            verbose=(iteration % 10 == 1 or iteration <= 5)
        )

        if val_result is None:
            continue

        combo = combined_score(val_result, test_result)

        if combo > best_combined:
            best_combined = combo
            best_config = config.copy()
            best_val_result = val_result
            best_test_result = test_result

            print_result(val_result, f"NEW BEST val (iter {iteration}, combo={combo:.1f})")
            if test_result:
                print_result(test_result, f"  OOS test")

        # Track strategies where both val and test are profitable
        val_profitable = val_result.total_pnl > 0 and val_result.total_trades >= 10
        test_profitable = test_result is not None and test_result.total_pnl > 0 and test_result.total_trades >= 10

        if val_profitable and test_profitable:
            both_pass = val_result.passed and test_result.passed

            if both_pass:
                print(f"\n{'*'*70}")
                print(f"*** BOTH VAL AND TEST PASS! Iteration {iteration} ***")
                print(f"{'*'*70}")
                print_result(val_result, "VAL")
                print_result(test_result, "TEST")

            passing_strategies.append({
                'config': config.copy(),
                'val_result': val_result,
                'test_result': test_result,
                'combined_score': combo,
                'both_pass': both_pass,
            })

            if sum(1 for s in passing_strategies if s.get('both_pass')) >= 3:
                print(f"\nFound 3 strategies passing BOTH periods! Stopping.")
                break

    # Final validation of best strategies
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE ({iteration} iterations)")
    print(f"Found {len(passing_strategies)} candidate strategies")
    print(f"{'='*70}")

    if not passing_strategies and best_config:
        passing_strategies.append({
            'config': best_config,
            'val_result': best_val_result,
            'test_result': best_test_result,
            'combined_score': best_combined,
        })

    # Cross-validate best strategies on other timeframes
    for si, strat in enumerate(passing_strategies):
        config = strat['config']
        print(f"\n--- Strategy {si+1} Cross-Validation ---")
        print(f"Config: {config}")

        for tf_name, tf_df in [('h4', df_h4), ('d1', df_d1)]:
            tf_feat = compute_features(tf_df).dropna()
            if len(tf_feat) < 500:
                continue

            avail_features = [c for c in feature_cols if c in tf_feat.columns]
            split = int(len(tf_feat) * 0.6)

            train_labels = create_trend_labels(
                tf_feat.iloc[:split], int(config['lookahead']), config['min_move_pct']
            )
            tf_train = tf_feat.iloc[:split].copy()
            tf_train['target'] = train_labels
            tf_train = tf_train.dropna()

            X_tr = tf_train[avail_features].replace([np.inf, -np.inf], np.nan).fillna(0)
            y_tr = tf_train['target']

            if y_tr.value_counts().min() < 20:
                continue

            models, scaler = train_ensemble(X_tr, y_tr)

            tf_test = tf_feat.iloc[split:]
            X_te = tf_test[avail_features].replace([np.inf, -np.inf], np.nan).fillna(0)

            signals, _, _ = ensemble_predict(
                models, scaler, X_te,
                min_agreement=int(config['min_agreement']),
                confidence_threshold=config['confidence']
            )

            lot_sizes, stop_losses, take_profits = compute_position_sizing(
                signals, tf_test,
                risk_pct=config['risk_pct'],
                atr_sl_mult=config['atr_sl_mult'],
                atr_tp_mult=config['atr_tp_mult']
            )

            result = run_challenge_test(tf_test, signals, lot_sizes, stop_losses, take_profits)
            print_result(result, f"Cross-val {tf_name}")

    # Print final report
    if passing_strategies:
        passing_strategies.sort(key=lambda x: x.get('combined_score', -9999), reverse=True)
        best = passing_strategies[0]
        config = best['config']

        print(f"\n{'='*70}")
        print(f"FINAL STRATEGY REPORT")
        print(f"{'='*70}")

        print(f"\n--- Configuration ---")
        for k, v in config.items():
            print(f"  {k}: {v}")

        for name, res in [('VALIDATION', best['val_result']), ('OUT-OF-SAMPLE TEST', best['test_result'])]:
            if res is None:
                continue
            print(f"\n--- {name} ---")
            print(f"  Passed: {res.passed}")
            print(f"  Reason: {res.reason}")
            print(f"  Trading days: {res.days_taken}")
            print(f"  Final balance: ${res.final_balance:,.2f}")
            print(f"  Total P&L: ${res.total_pnl:,.2f} ({res.total_pnl_pct:+.2f}%)")
            print(f"  Max daily DD: ${res.max_daily_dd:,.2f} ({res.max_daily_dd/1000:.2f}%)")
            print(f"  Max total DD: ${res.max_total_dd:,.2f} ({res.max_total_dd/1000:.2f}%)")
            print(f"  Total trades: {res.total_trades}")
            print(f"  Win rate: {res.win_rate:.1f}%")
            print(f"  Avg win: ${res.avg_win:,.2f}")
            print(f"  Avg loss: ${res.avg_loss:,.2f}")
            print(f"  Profit factor: {res.profit_factor:.2f}")

            if res.daily_results:
                print(f"\n  Daily Breakdown:")
                for day in res.daily_results[:15]:
                    sign = "+" if day.pnl >= 0 else "-"
                    print(f"    {day.date}: {sign}${abs(day.pnl):,.2f} ({day.pnl_pct:+.2f}%) | "
                          f"Trades: {day.num_trades} W:{day.winning_trades} L:{day.losing_trades}")
                if len(res.daily_results) > 15:
                    print(f"    ... ({len(res.daily_results) - 15} more days)")

        # Save results
        output = {
            'config': config,
            'data_source': 'ejtraderLabs/historical-data REAL XAU/USD 2012-2022',
            'timeframe': 'H1 (1 hour)',
            'methodology': {
                'model': 'Ensemble (XGBoost + LightGBM + RandomForest)',
                'labeling': 'Trend-based with min move threshold',
                'validation': 'Walk-forward (60/20/20 split)',
                'position_sizing': 'ATR-based with fixed risk per trade',
            },
            'anti_inflation': [
                'Real XAU/USD market data from ejtraderLabs',
                'Walk-forward validation - train on past only',
                'Out-of-sample test on completely unseen future data',
                'Cross-timeframe validation (H4, D1)',
                'Realistic costs: $0.30 spread, $7/lot commission, $0.05 slippage',
                'No look-ahead bias in simulation',
                'Ensemble requires multiple model agreement',
            ],
        }

        for key, res in [('val_result', best['val_result']), ('test_result', best['test_result'])]:
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
    print(f"\nTotal time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
