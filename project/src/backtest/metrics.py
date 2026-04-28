from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame, curve: pd.DataFrame) -> dict:
    if curve.empty:
        return {}
    curve = curve.copy()
    curve["ret"] = curve["equity"].pct_change().fillna(0)
    ann = 365 * 24 * 60
    mu = curve["ret"].mean() * ann
    vol = curve["ret"].std() * np.sqrt(ann)
    neg = curve.loc[curve["ret"] < 0, "ret"].std() * np.sqrt(ann)
    sharpe = mu / vol if vol else 0.0
    sortino = mu / neg if neg else 0.0
    rolling_max = curve["equity"].cummax()
    dd = (curve["equity"] - rolling_max) / rolling_max.replace(0, np.nan)
    max_dd = abs(dd.min()) if len(dd) else 0
    total_return = curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1
    years = max((curve["timestamp"].iloc[-1] - curve["timestamp"].iloc[0]).total_seconds() / (365 * 24 * 3600), 1e-8)
    cagr = (1 + total_return) ** (1 / years) - 1
    calmar = cagr / max_dd if max_dd else 0.0

    win = (trades["pnl"] > 0).mean() if not trades.empty else 0.0
    gross_win = trades.loc[trades["pnl"] > 0, "pnl"].sum() if not trades.empty else 0.0
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum() if not trades.empty else 0.0
    profit_factor = gross_win / gross_loss if gross_loss else 0.0
    expectancy = trades["pnl"].mean() if not trades.empty else 0.0
    avg_r = trades["r_multiple"].mean() if (not trades.empty and "r_multiple" in trades.columns) else 0.0

    return {
        "CAGR": float(cagr),
        "total_return": float(total_return),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "Calmar": float(calmar),
        "win_rate": float(win),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "average_R": float(avg_r),
        "exposure": 1.0,
        "turnover": float(len(trades)),
    }


def save_metrics(metrics: dict, out_file: Path) -> None:
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
