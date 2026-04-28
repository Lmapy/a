#!/usr/bin/env python3
"""Backtest: trade each 4h candle in previous 4h candle direction (stdlib only)."""
import argparse
import csv
import datetime as dt
import math
from pathlib import Path


def floor_4h(ts: dt.datetime) -> dt.datetime:
    hour = (ts.hour // 4) * 4
    return ts.replace(hour=hour, minute=0, second=0, microsecond=0)


def load_1h_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = dt.datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
            rows.append(
                {
                    "timestamp": ts.astimezone(dt.timezone.utc),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                }
            )
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def resample_4h(rows_1h: list[dict]) -> list[dict]:
    buckets: dict[dt.datetime, list[dict]] = {}
    for r in rows_1h:
        key = floor_4h(r["timestamp"])
        buckets.setdefault(key, []).append(r)

    bars = []
    for key in sorted(buckets):
        b = buckets[key]
        bars.append(
            {
                "timestamp": key,
                "open": b[0]["open"],
                "high": max(x["high"] for x in b),
                "low": min(x["low"] for x in b),
                "close": b[-1]["close"],
            }
        )
    return bars


def backtest(bars: list[dict]) -> list[dict]:
    trades = []
    equity = 1.0
    benchmark = 1.0
    for i in range(1, len(bars)):
        prev = bars[i - 1]
        cur = bars[i]
        prev_dir = 1 if prev["close"] > prev["open"] else -1 if prev["close"] < prev["open"] else 0
        if prev_dir == 0:
            continue
        raw_ret = cur["close"] / cur["open"] - 1.0
        strat_ret = prev_dir * raw_ret
        equity *= 1.0 + strat_ret
        benchmark *= 1.0 + raw_ret
        trades.append(
            {
                "timestamp": cur["timestamp"].isoformat(),
                "signal": prev_dir,
                "open": cur["open"],
                "close": cur["close"],
                "bar_return": raw_ret,
                "strategy_return": strat_ret,
                "equity_curve": equity,
                "benchmark_curve": benchmark,
            }
        )
    return trades


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"bars": 0}

    rets = [t["strategy_return"] for t in trades]
    wins = sum(1 for r in rets if r > 0)
    avg = sum(rets) / n
    variance = sum((r - avg) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    sharpe = (avg / std * math.sqrt(6 * 252)) if std > 0 else 0.0

    peak = -1.0
    max_dd = 0.0
    for t in trades:
        eq = t["equity_curve"]
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "bars": n,
        "start": trades[0]["timestamp"],
        "end": trades[-1]["timestamp"],
        "win_rate": wins / n,
        "avg_return_per_trade": avg,
        "total_return": trades[-1]["equity_curve"] - 1.0,
        "benchmark_return": trades[-1]["benchmark_curve"] - 1.0,
        "annualized_sharpe_approx": sharpe,
        "max_drawdown": max_dd,
    }


def write_trades(path: Path, trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "timestamp",
            "signal",
            "open",
            "close",
            "bar_return",
            "strategy_return",
            "equity_curve",
            "benchmark_curve",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trades)


def write_report(path: Path, s: dict) -> None:
    lines = [
        "# Gold 4H Continuation Backtest Report",
        "",
        "## Hypothesis",
        "Go long/short at each 4H open in the same direction as the previous 4H candle body.",
        "",
        "## Results",
    ]
    for k, v in s.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Notes",
        "- No transaction costs or slippage included.",
        "- Baseline directional persistence test only.",
        "- Next: add costs, regime filters, and walk-forward validation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-trades", default="results/gold_4h_trades.csv")
    p.add_argument("--output-report", default="results/backtest_report.md")
    args = p.parse_args()

    rows_1h = load_1h_rows(Path(args.input))
    bars_4h = resample_4h(rows_1h)
    trades = backtest(bars_4h)
    summary = summarize(trades)

    write_trades(Path(args.output_trades), trades)
    write_report(Path(args.output_report), summary)

    print("Backtest complete")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
