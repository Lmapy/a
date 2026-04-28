from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir


@dataclass
class Position:
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop: float
    target: float | None
    tag: str


def _risk_position_size(equity: float, risk_pct: float, entry: float, stop: float, leverage_cap: float) -> float:
    risk_dollar = equity * risk_pct
    stop_dist = max(abs(entry - stop), 1e-8)
    qty = risk_dollar / stop_dist
    notional = qty * entry
    max_notional = equity * leverage_cap
    if notional > max_notional:
        qty *= max_notional / notional
    return qty


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, cfg: dict, strategy_name: str, out_dir: Path) -> dict:
    fee = cfg["execution"]["taker_fee_bps"] / 1e4
    slip = cfg["execution"]["slippage_bps"] / 1e4
    risk_pct = cfg["execution"]["risk_per_trade"]
    leverage_cap = cfg["execution"]["leverage_cap"]
    max_pos = cfg["execution"]["max_concurrent_positions"]

    equity = cfg["execution"]["initial_capital"]
    start_equity = equity
    positions: list[Position] = []
    trades: list[dict] = []
    curve: list[dict] = []
    max_equity = equity
    halt = False

    merged = df.merge(signals[["timestamp", "long_entry", "short_entry", "stop_price", "target_price", "tag"]], on="timestamp", how="left")
    merged = merged.fillna({"long_entry": False, "short_entry": False})

    for row in merged.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        px = float(row.close)
        funding = float(row.funding_rate) if pd.notna(row.funding_rate) else 0.0

        unreal = 0.0
        for p in positions:
            unreal += p.side * (px - p.entry_price) * p.qty
        curve.append({"timestamp": ts, "equity": equity + unreal})

        if halt:
            continue

        still_open: list[Position] = []
        for p in positions:
            stop_hit = (p.side == 1 and row.low <= p.stop) or (p.side == -1 and row.high >= p.stop)
            target_hit = p.target is not None and ((p.side == 1 and row.high >= p.target) or (p.side == -1 and row.low <= p.target))
            if stop_hit or target_hit:
                exit_px = p.stop if stop_hit else p.target
                assert exit_px is not None
                pnl = p.side * (exit_px - p.entry_price) * p.qty
                fees = (abs(p.entry_price * p.qty) + abs(exit_px * p.qty)) * fee
                funding_pnl = -p.side * funding * p.entry_price * p.qty
                net = pnl - fees + funding_pnl
                equity += net
                trades.append(
                    {
                        "strategy": strategy_name,
                        "entry_time": p.entry_time,
                        "exit_time": ts,
                        "side": p.side,
                        "entry_price": p.entry_price,
                        "exit_price": exit_px,
                        "qty": p.qty,
                        "pnl": net,
                        "r_multiple": net / max(abs((p.entry_price - p.stop) * p.qty), 1e-8),
                        "tag": p.tag,
                    }
                )
            else:
                still_open.append(p)
        positions = still_open

        max_equity = max(max_equity, equity)
        drawdown = (max_equity - equity) / max_equity if max_equity else 0
        if drawdown >= cfg["execution"]["max_drawdown_kill_switch_pct"]:
            halt = True
            continue

        if len(positions) >= max_pos:
            continue

        if bool(row.long_entry) or bool(row.short_entry):
            side = 1 if bool(row.long_entry) else -1
            entry_px = px * (1 + slip * side)
            stop = float(row.stop_price) if pd.notna(row.stop_price) else px - side * float(row.atr_14)
            tgt = float(row.target_price) if pd.notna(row.target_price) else None
            qty = _risk_position_size(equity, risk_pct, entry_px, stop, leverage_cap)
            positions.append(Position(side=side, entry_time=ts, entry_price=entry_px, qty=qty, stop=stop, target=tgt, tag=str(row.tag)))

    trade_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)
    ensure_dir(out_dir)
    trade_df.to_parquet(out_dir / f"{strategy_name}_trade_log.parquet", index=False)
    trade_df.to_csv(out_dir / f"{strategy_name}_trade_log.csv", index=False)
    curve_df.to_parquet(out_dir / f"{strategy_name}_equity_curve.parquet", index=False)
    curve_df.to_csv(out_dir / f"{strategy_name}_equity_curve.csv", index=False)

    total_return = (equity / start_equity) - 1
    return {
        "strategy": strategy_name,
        "total_return": float(total_return),
        "ending_equity": float(equity),
        "num_trades": int(len(trade_df)),
    }
