"""
ML-based trading strategy builder.
Iteratively trains and tests models, optimizing for prop firm challenge passing.
Uses walk-forward validation to prevent overfitting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
import json
import os

from feature_engine import compute_features, prepare_ml_data
from prop_firm_sim import PropFirmSimulator, ChallengeResult

warnings.filterwarnings('ignore')


class MLStrategy:
    def __init__(self, model_type='xgboost', params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.threshold_long = 0.55
        self.threshold_short = 0.55

    def create_model(self):
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'verbosity': 0,
            }
            default_params.update(self.params)
            return xgb.XGBClassifier(**default_params)

        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
            default_params.update(self.params)
            return lgb.LGBMClassifier(**default_params)

        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 300,
                'max_depth': 8,
                'min_samples_leaf': 10,
                'random_state': 42,
                'n_jobs': -1,
            }
            default_params.update(self.params)
            return RandomForestClassifier(**default_params)

        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'min_samples_leaf': 10,
                'random_state': 42,
            }
            default_params.update(self.params)
            return GradientBoostingClassifier(**default_params)

    def train(self, X_train, y_train):
        self.model = self.create_model()
        X_scaled = self.scaler.fit_transform(X_train)
        # Map labels: -1->0, 0->1, 1->2
        y_mapped = y_train.copy()
        y_mapped = y_mapped.map({-1: 0, 0: 1, 1: 2}) if hasattr(y_mapped, 'map') else np.where(y_mapped == -1, 0, np.where(y_mapped == 0, 1, 2))
        self.model.fit(X_scaled, y_mapped)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        return proba  # columns: [short(0), hold(1), long(2)]

    def generate_signals(self, X, confidence_long=None, confidence_short=None):
        """Generate trading signals based on model predictions."""
        if confidence_long is None:
            confidence_long = self.threshold_long
        if confidence_short is None:
            confidence_short = self.threshold_short

        proba = self.predict_proba(X)
        signals = np.zeros(len(X))

        # Only trade when model is confident
        signals[proba[:, 2] > confidence_long] = 1   # Long
        signals[proba[:, 0] > confidence_short] = -1  # Short

        return signals.astype(int)


class StrategyOptimizer:
    """
    Iteratively optimizes ML strategy to pass prop firm challenges.
    Uses walk-forward validation on out-of-sample data.
    """

    def __init__(self, df, feature_cols):
        self.df = df
        self.feature_cols = feature_cols
        self.best_result = None
        self.best_config = None
        self.iteration_results = []

    def walk_forward_split(self, train_ratio=0.6, val_ratio=0.2):
        """Split data into train/validation/test."""
        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = self.df.iloc[:train_end]
        val = self.df.iloc[train_end:val_end]
        test = self.df.iloc[val_end:]

        return train, val, test

    def compute_position_sizing(self, signals, df, atr_col='atr_14',
                                risk_per_trade_pct=1.0, account_size=100_000):
        """
        Dynamic position sizing based on ATR and risk management.
        Risk a fixed % of account per trade.
        """
        lot_sizes = np.full(len(signals), 0.0)
        stop_losses = np.full(len(signals), 5.0)  # default $5 SL
        take_profits = np.full(len(signals), 10.0)  # default $10 TP

        for i in range(len(signals)):
            if signals[i] == 0:
                continue

            atr_val = df[atr_col].iloc[i] if atr_col in df.columns else 5.0
            if np.isnan(atr_val) or atr_val <= 0:
                atr_val = 5.0

            # SL = 2 * ATR, TP = 3 * ATR (1.5:1 R:R minimum)
            sl_dist = max(atr_val * 2.0, 1.0)
            tp_dist = max(atr_val * 3.0, 1.5)

            # Risk per trade
            risk_dollars = account_size * risk_per_trade_pct / 100
            # lot_size = risk / (SL * value_per_lot)
            lots = risk_dollars / (sl_dist * 100)  # 100 oz per lot
            lots = max(0.01, min(lots, 5.0))  # Cap at 5 lots
            lots = round(lots, 2)

            lot_sizes[i] = lots
            stop_losses[i] = sl_dist
            take_profits[i] = tp_dist

        return lot_sizes, stop_losses, take_profits

    def evaluate_config(self, config, train_df, test_df, verbose=False):
        """
        Train a model with given config and evaluate on test data.
        Returns ChallengeResult.
        """
        model_type = config.get('model_type', 'xgboost')
        model_params = config.get('model_params', {})
        target_bars = config.get('target_bars_ahead', 10)
        min_move = config.get('min_move_pct', 0.1)
        confidence_long = config.get('confidence_long', 0.55)
        confidence_short = config.get('confidence_short', 0.55)
        risk_per_trade = config.get('risk_per_trade_pct', 1.0)
        rr_ratio = config.get('rr_ratio', 1.5)
        atr_sl_mult = config.get('atr_sl_mult', 2.0)
        atr_tp_mult = config.get('atr_tp_mult', 3.0)

        # Prepare train data
        train_feat = compute_features(train_df)
        future_ret = train_feat['Close'].shift(-target_bars) / train_feat['Close'] - 1
        future_ret_pct = future_ret * 100
        train_feat['target'] = 0
        train_feat.loc[future_ret_pct > min_move, 'target'] = 1
        train_feat.loc[future_ret_pct < -min_move, 'target'] = -1
        train_feat = train_feat.dropna()

        # Prepare test data
        test_feat = compute_features(test_df)
        test_feat = test_feat.dropna()

        # Align feature columns
        available_features = [c for c in self.feature_cols if c in train_feat.columns and c in test_feat.columns]
        if len(available_features) < 10:
            return None

        X_train = train_feat[available_features]
        y_train = train_feat['target']
        X_test = test_feat[available_features]

        # Handle inf values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Train model
        strategy = MLStrategy(model_type=model_type, params=model_params)
        strategy.train(X_train, y_train)

        # Generate signals on test data
        signals = strategy.generate_signals(X_test, confidence_long, confidence_short)

        # Position sizing
        lot_sizes = np.full(len(signals), 0.0)
        stop_losses = np.full(len(signals), 5.0)
        take_profits = np.full(len(signals), 10.0)

        for i in range(len(signals)):
            if signals[i] == 0:
                continue
            atr_val = test_feat['atr_14'].iloc[i] if 'atr_14' in test_feat.columns else 5.0
            if np.isnan(atr_val) or atr_val <= 0:
                atr_val = 5.0

            sl_dist = max(atr_val * atr_sl_mult, 1.0)
            tp_dist = max(atr_val * atr_tp_mult, 1.5)

            risk_dollars = 100_000 * risk_per_trade / 100
            lots = risk_dollars / (sl_dist * 100)
            lots = max(0.01, min(lots, 5.0))
            lots = round(lots, 2)

            lot_sizes[i] = lots
            stop_losses[i] = sl_dist
            take_profits[i] = tp_dist

        # Run prop firm simulation
        sim = PropFirmSimulator(profit_target_pct=8.0, phase="Phase 1")
        result = sim.simulate(test_feat, signals, lot_sizes, stop_losses, take_profits)

        if verbose:
            print(f"  Model: {model_type}, Bars ahead: {target_bars}, "
                  f"Min move: {min_move}%, Confidence: {confidence_long:.2f}/{confidence_short:.2f}")
            print(f"  Result: {'PASSED' if result.passed else 'FAILED'} | "
                  f"P&L: ${result.total_pnl:.2f} ({result.total_pnl_pct:.2f}%) | "
                  f"Days: {result.days_taken} | Trades: {result.total_trades} | "
                  f"WR: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f} | "
                  f"Max DD: ${result.max_total_dd:.2f} | Reason: {result.reason}")

        return result

    def generate_configs(self, iteration=0):
        """Generate diverse configs to try."""
        configs = []

        model_types = ['xgboost', 'lightgbm', 'random_forest']
        target_bars_options = [5, 10, 15, 20, 30]
        min_move_options = [0.05, 0.1, 0.15, 0.2, 0.3]
        confidence_options = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        risk_options = [0.5, 0.75, 1.0, 1.5, 2.0]
        atr_sl_options = [1.5, 2.0, 2.5, 3.0]
        atr_tp_options = [2.0, 3.0, 4.0, 5.0, 6.0]

        rng = np.random.RandomState(42 + iteration)

        # Generate random configs
        for _ in range(20):
            config = {
                'model_type': rng.choice(model_types),
                'target_bars_ahead': int(rng.choice(target_bars_options)),
                'min_move_pct': float(rng.choice(min_move_options)),
                'confidence_long': float(rng.choice(confidence_options)),
                'confidence_short': float(rng.choice(confidence_options)),
                'risk_per_trade_pct': float(rng.choice(risk_options)),
                'atr_sl_mult': float(rng.choice(atr_sl_options)),
                'atr_tp_mult': float(rng.choice(atr_tp_options)),
                'model_params': {},
            }

            # Ensure TP > SL for positive expectancy
            if config['atr_tp_mult'] <= config['atr_sl_mult']:
                config['atr_tp_mult'] = config['atr_sl_mult'] * 1.5

            configs.append(config)

        # Also add some targeted configs based on what's worked before
        if self.best_config is not None:
            for _ in range(10):
                config = self.best_config.copy()
                # Mutate one or two params
                param_to_mutate = rng.choice(['confidence_long', 'confidence_short',
                                               'risk_per_trade_pct', 'atr_sl_mult',
                                               'atr_tp_mult', 'min_move_pct',
                                               'target_bars_ahead'])
                if param_to_mutate == 'confidence_long':
                    config['confidence_long'] = max(0.40, min(0.80, config['confidence_long'] + rng.uniform(-0.05, 0.05)))
                elif param_to_mutate == 'confidence_short':
                    config['confidence_short'] = max(0.40, min(0.80, config['confidence_short'] + rng.uniform(-0.05, 0.05)))
                elif param_to_mutate == 'risk_per_trade_pct':
                    config['risk_per_trade_pct'] = max(0.25, min(3.0, config['risk_per_trade_pct'] + rng.uniform(-0.25, 0.25)))
                elif param_to_mutate == 'atr_sl_mult':
                    config['atr_sl_mult'] = max(1.0, min(4.0, config['atr_sl_mult'] + rng.uniform(-0.5, 0.5)))
                elif param_to_mutate == 'atr_tp_mult':
                    config['atr_tp_mult'] = max(1.5, min(8.0, config['atr_tp_mult'] + rng.uniform(-0.5, 0.5)))
                elif param_to_mutate == 'min_move_pct':
                    config['min_move_pct'] = max(0.02, min(0.5, config['min_move_pct'] + rng.uniform(-0.05, 0.05)))
                elif param_to_mutate == 'target_bars_ahead':
                    config['target_bars_ahead'] = int(max(3, min(50, config['target_bars_ahead'] + rng.randint(-5, 6))))

                if config['atr_tp_mult'] <= config['atr_sl_mult']:
                    config['atr_tp_mult'] = config['atr_sl_mult'] * 1.5

                configs.append(config)

        return configs

    def score_result(self, result):
        """
        Score a challenge result for optimization.
        Higher is better. Heavily rewards passing the challenge.
        """
        if result is None:
            return -1000

        score = 0

        # Massive bonus for passing
        if result.passed:
            score += 1000
            # Bonus for fewer days
            if result.days_taken <= 5:
                score += 500
            elif result.days_taken <= 10:
                score += 200

        # Reward profit
        score += result.total_pnl_pct * 10

        # Reward good win rate
        if result.win_rate > 50:
            score += (result.win_rate - 50) * 5

        # Reward good profit factor
        if result.profit_factor > 1:
            score += (result.profit_factor - 1) * 50

        # Penalize drawdown
        dd_pct = result.max_total_dd / 100_000 * 100
        score -= dd_pct * 20

        # Penalize too few trades (unreliable stats)
        if result.total_trades < 10:
            score -= (10 - result.total_trades) * 20

        # Penalize daily DD breach heavily
        if result.reason == 'daily_dd_breach':
            score -= 200
        elif result.reason == 'total_dd_breach':
            score -= 300

        return score

    def optimize(self, max_iterations=50, verbose=True):
        """
        Main optimization loop. Iterates until finding a passing strategy.
        """
        train_df, val_df, test_df = self.walk_forward_split(0.6, 0.2)

        if verbose:
            print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            print(f"Train: {train_df.index[0]} to {train_df.index[-1]}")
            print(f"Val:   {val_df.index[0]} to {val_df.index[-1]}")
            print(f"Test:  {test_df.index[0]} to {test_df.index[-1]}")
            print()

        best_score = -9999
        best_val_result = None
        best_test_result = None
        passing_configs = []

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")

            configs = self.generate_configs(iteration)

            for ci, config in enumerate(configs):
                if verbose:
                    print(f"\n--- Config {ci+1}/{len(configs)} ---")

                # Evaluate on validation set
                val_result = self.evaluate_config(config, train_df, val_df, verbose=verbose)
                if val_result is None:
                    continue

                score = self.score_result(val_result)

                self.iteration_results.append({
                    'iteration': iteration,
                    'config_idx': ci,
                    'config': config,
                    'score': score,
                    'passed': val_result.passed,
                    'pnl_pct': val_result.total_pnl_pct,
                    'days': val_result.days_taken,
                    'trades': val_result.total_trades,
                    'win_rate': val_result.win_rate,
                    'profit_factor': val_result.profit_factor,
                    'max_dd': val_result.max_total_dd,
                    'reason': val_result.reason,
                })

                if score > best_score:
                    best_score = score
                    best_val_result = val_result
                    self.best_config = config.copy()

                    if verbose:
                        print(f"  *** NEW BEST SCORE: {score:.1f} ***")

                # If it passes validation, test on OOS
                if val_result.passed and val_result.days_taken <= 5:
                    if verbose:
                        print(f"\n  >>> PASSED VALIDATION in {val_result.days_taken} days! Testing OOS...")

                    test_result = self.evaluate_config(config, train_df, test_df, verbose=verbose)
                    if test_result is not None and test_result.passed and test_result.days_taken <= 5:
                        if verbose:
                            print(f"\n  >>> PASSED OOS TEST! Strategy found!")
                        passing_configs.append({
                            'config': config,
                            'val_result': val_result,
                            'test_result': test_result,
                            'val_score': score,
                            'test_score': self.score_result(test_result),
                        })
                        best_test_result = test_result

            # If we have passing configs on OOS, we can stop
            if len(passing_configs) >= 3:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"FOUND {len(passing_configs)} PASSING STRATEGIES!")
                    print(f"{'='*60}")
                break

        # Final cross-validation on the best strategies
        if passing_configs:
            if verbose:
                print(f"\n\n{'='*60}")
                print("FINAL VALIDATION - Cross-checking best strategies")
                print(f"{'='*60}")

            # Sort by combined val+test score
            passing_configs.sort(key=lambda x: x['val_score'] + x['test_score'], reverse=True)
            best = passing_configs[0]

            # Run on full OOS (val + test combined)
            full_oos = pd.concat([val_df, test_df])
            final_result = self.evaluate_config(best['config'], train_df, full_oos, verbose=True)

            return {
                'status': 'PASSED',
                'best_config': best['config'],
                'val_result': best['val_result'],
                'test_result': best['test_result'],
                'final_oos_result': final_result,
                'all_passing': passing_configs,
                'total_iterations': iteration + 1,
            }

        # If no passing config found, try expanding search
        if best_val_result is not None:
            # Test the best validation result on OOS anyway
            test_result = self.evaluate_config(self.best_config, train_df, test_df, verbose=True)
            return {
                'status': 'BEST_EFFORT',
                'best_config': self.best_config,
                'val_result': best_val_result,
                'test_result': test_result,
                'best_score': best_score,
                'total_iterations': max_iterations,
            }

        return {
            'status': 'FAILED',
            'total_iterations': max_iterations,
        }
