from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics, save_metrics
from src.backtest.plotting import make_charts
from src.strategies.breakout import generate_breakout_signals
from src.strategies.sweep import generate_sweep_signals
from src.strategies.vwap_reversion import generate_vwap_reversion_signals


def _strategy_signals(strategy: str, feat_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if strategy == "breakout":
        return generate_breakout_signals(feat_df, cfg["strategies"]["breakout"])
    if strategy == "sweep":
        return generate_sweep_signals(feat_df, cfg["strategies"]["sweep"])
    if strategy == "vwap_reversion":
        return generate_vwap_reversion_signals(feat_df, cfg["strategies"]["vwap_reversion"])
    raise ValueError(f"unknown strategy {strategy}")


def run_strategy_backtest(config: dict, strategy: str) -> None:
    proc_dir = Path(config["project"]["processed_data_dir"])
    out_dir = proc_dir / "backtests" / strategy
    frames: list[pd.DataFrame] = []

    for exchange in config["symbols"].keys():
        ex_dir = proc_dir / exchange
        if not ex_dir.exists():
            continue
        for p in ex_dir.glob("*_features_1m.parquet"):
            df = pd.read_parquet(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames.append(df)

    if not frames:
        raise RuntimeError("No processed feature files found. Run features command first.")

    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    signals = _strategy_signals(strategy, data, config)
    run_backtest(data, signals, config, strategy, out_dir)

    trades = pd.read_parquet(out_dir / f"{strategy}_trade_log.parquet")
    curve = pd.read_parquet(out_dir / f"{strategy}_equity_curve.parquet")
    metrics = compute_metrics(trades, curve)
    save_metrics(metrics, out_dir / f"{strategy}_summary_metrics.json")
    make_charts(trades, curve, out_dir, strategy)


def run_portfolio_backtest(config: dict) -> None:
    for strategy in ["breakout", "sweep", "vwap_reversion"]:
        run_strategy_backtest(config, strategy)
