from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_charts(trades: pd.DataFrame, curve: pd.DataFrame, out_dir: Path, strategy_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = curve.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)

    plt.figure(figsize=(10, 4))
    plt.plot(curve["timestamp"], curve["equity"])
    plt.title(f"Equity Curve - {strategy_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{strategy_name}_equity_curve.png")
    plt.close()

    roll_max = curve["equity"].cummax()
    dd = (curve["equity"] - roll_max) / roll_max
    plt.figure(figsize=(10, 4))
    plt.plot(curve["timestamp"], dd)
    plt.title(f"Drawdown - {strategy_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{strategy_name}_drawdown.png")
    plt.close()

    monthly = curve.set_index("timestamp")["equity"].resample("M").last().pct_change().dropna()
    plt.figure(figsize=(10, 4))
    plt.bar(monthly.index.astype(str), monthly.values)
    plt.xticks(rotation=60, fontsize=7)
    plt.title(f"Monthly Returns - {strategy_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{strategy_name}_monthly_returns.png")
    plt.close()

    if not trades.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(trades["pnl"], bins=40)
        plt.title(f"Trade Return Distribution - {strategy_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{strategy_name}_trade_distribution.png")
        plt.close()
