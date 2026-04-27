#!/usr/bin/env python3
"""Walk-forward optimizer for gold 4H candle-direction strategy.

Goal: find parameter sets that keep trading cadence near a target range
(default 3-5 trades/week) while maximizing risk-adjusted returns.
"""

import argparse
import csv
import datetime as dt
import itertools
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


def parse_hours_csv(v: str) -> list[int]:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def compute_week_span(start_ts: str, end_ts: str) -> float:
    start = dt.datetime.fromisoformat(start_ts)
    end = dt.datetime.fromisoformat(end_ts)
    weeks = (end - start).total_seconds() / (7 * 24 * 3600)
    return max(weeks, 1.0 / 7.0)


def run_strategy(bars: list[dict], params: dict) -> list[dict]:
    trades = []
    equity = 1.0
    for i in range(1, len(bars)):
        prev = bars[i - 1]
        cur = bars[i]
        cur_hour = cur["timestamp"].hour
        if cur_hour not in params["entry_hours"]:
            continue
        if cur["timestamp"].weekday() not in params["entry_weekdays"]:
            continue

        prev_body = (prev["close"] - prev["open"]) / prev["open"]
        if abs(prev_body) < params["min_prev_body_pct"]:
            continue

        signal = 1 if prev_body > 0 else -1
        if params["direction_mode"] == "reversal":
            signal *= -1

        ret = cur["close"] / cur["open"] - 1.0
        strat_ret = signal * ret
        strat_ret -= params["fee_per_trade"]

        equity *= 1.0 + strat_ret
        trades.append(
            {
                "timestamp": cur["timestamp"].isoformat(),
                "signal": signal,
                "open": cur["open"],
                "close": cur["close"],
                "prev_body_pct": prev_body,
                "strategy_return": strat_ret,
                "equity_curve": equity,
            }
        )
    return trades


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "trades_per_week": 0.0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "sharpe_approx": 0.0,
            "max_drawdown": 0.0,
            "start": "n/a",
            "end": "n/a",
        }

    n = len(trades)
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

    weeks = compute_week_span(trades[0]["timestamp"], trades[-1]["timestamp"])

    return {
        "trades": n,
        "trades_per_week": n / weeks,
        "win_rate": wins / n,
        "avg_return": avg,
        "total_return": trades[-1]["equity_curve"] - 1.0,
        "sharpe_approx": sharpe,
        "max_drawdown": max_dd,
        "start": trades[0]["timestamp"],
        "end": trades[-1]["timestamp"],
    }


def score_summary(summary: dict, min_tpw: float, max_tpw: float) -> float:
    tpw = summary["trades_per_week"]
    if tpw < min_tpw:
        penalty = (min_tpw - tpw) * 8.0
    elif tpw > max_tpw:
        penalty = (tpw - max_tpw) * 8.0
    else:
        penalty = 0.0

    return summary["sharpe_approx"] + summary["total_return"] * 10.0 - penalty


def generate_candidates(hours_universe: list[int], min_body_grid: list[float], fee_grid: list[float]) -> list[dict]:
    candidates = []
    hour_sets = [
        tuple(hours_universe),
        tuple(h for h in hours_universe if h in (0, 8, 16)),
        tuple(h for h in hours_universe if h in (0, 4, 8, 12)),
        tuple(h for h in hours_universe if h in (8, 12, 16, 20)),
        tuple(h for h in hours_universe if h in (8,)),
        tuple(h for h in hours_universe if h in (12,)),
    ]
    hour_sets = sorted(set(hour_sets))
    weekday_sets = [
        (0, 1, 2, 3, 4),  # Mon-Fri
        (0, 2, 4),  # Mon/Wed/Fri
        (1, 3),  # Tue/Thu
    ]

    for direction_mode, entry_hours, entry_weekdays, min_prev_body_pct, fee in itertools.product(
        ["continuation", "reversal"], hour_sets, weekday_sets, min_body_grid, fee_grid
    ):
        candidates.append(
            {
                "direction_mode": direction_mode,
                "entry_hours": list(entry_hours),
                "entry_weekdays": list(entry_weekdays),
                "min_prev_body_pct": min_prev_body_pct,
                "fee_per_trade": fee,
            }
        )
    return candidates


def walkforward(
    bars: list[dict],
    train_days: int,
    test_days: int,
    min_tpw: float,
    max_tpw: float,
    candidates: list[dict],
    min_train_bars: int,
    min_test_bars: int,
) -> list[dict]:
    if not bars:
        return []

    folds = []
    start_ts = bars[0]["timestamp"]
    end_ts = bars[-1]["timestamp"]
    cursor = start_ts + dt.timedelta(days=train_days)

    while cursor + dt.timedelta(days=test_days) <= end_ts:
        train_start = cursor - dt.timedelta(days=train_days)
        train_end = cursor
        test_end = cursor + dt.timedelta(days=test_days)

        train_bars = [b for b in bars if train_start <= b["timestamp"] < train_end]
        test_bars = [b for b in bars if train_end <= b["timestamp"] < test_end]

        if len(train_bars) < min_train_bars or len(test_bars) < min_test_bars:
            cursor += dt.timedelta(days=test_days)
            continue

        best_params = None
        best_score = -10**9
        best_train_summary = None

        for params in candidates:
            train_trades = run_strategy(train_bars, params)
            train_summary = summarize(train_trades)
            score = score_summary(train_summary, min_tpw=min_tpw, max_tpw=max_tpw)
            if score > best_score:
                best_score = score
                best_params = params
                best_train_summary = train_summary

        test_trades = run_strategy(test_bars, best_params)
        test_summary = summarize(test_trades)

        folds.append(
            {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": train_end.isoformat(),
                "test_end": test_end.isoformat(),
                "best_params": best_params,
                "train_summary": best_train_summary,
                "test_summary": test_summary,
                "selection_score": best_score,
            }
        )

        cursor += dt.timedelta(days=test_days)

    return folds


def aggregate_oos(folds: list[dict], min_tpw: float, max_tpw: float) -> dict:
    if not folds:
        return {"folds": 0}

    oos_trades = [f["test_summary"]["trades"] for f in folds]
    oos_tpw = [f["test_summary"]["trades_per_week"] for f in folds]
    oos_ret = [f["test_summary"]["total_return"] for f in folds]

    return {
        "folds": len(folds),
        "avg_oos_trades_per_fold": sum(oos_trades) / len(oos_trades),
        "avg_oos_trades_per_week": sum(oos_tpw) / len(oos_tpw),
        "avg_oos_total_return_per_fold": sum(oos_ret) / len(oos_ret),
        "oos_trade_frequency_target_met_ratio": sum(1 for x in oos_tpw if min_tpw <= x <= max_tpw) / len(oos_tpw),
    }


def write_folds_csv(path: Path, folds: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "direction_mode",
            "entry_hours",
            "entry_weekdays",
            "min_prev_body_pct",
            "fee_per_trade",
            "train_trades_per_week",
            "test_trades_per_week",
            "test_total_return",
            "test_sharpe_approx",
            "selection_score",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fold in folds:
            params = fold["best_params"]
            train = fold["train_summary"]
            test = fold["test_summary"]
            w.writerow(
                {
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "direction_mode": params["direction_mode"],
                    "entry_hours": ",".join(str(x) for x in params["entry_hours"]),
                    "entry_weekdays": ",".join(str(x) for x in params["entry_weekdays"]),
                    "min_prev_body_pct": params["min_prev_body_pct"],
                    "fee_per_trade": params["fee_per_trade"],
                    "train_trades_per_week": train["trades_per_week"],
                    "test_trades_per_week": test["trades_per_week"],
                    "test_total_return": test["total_return"],
                    "test_sharpe_approx": test["sharpe_approx"],
                    "selection_score": fold["selection_score"],
                }
            )


def write_report(path: Path, folds: list[dict], agg: dict, min_tpw: float, max_tpw: float) -> None:
    lines = [
        "# Walk-Forward Optimization Report (Gold 4H Direction Model)",
        "",
        "## Objective",
        f"Optimize parameters with an explicit trade-frequency target of **{min_tpw:.1f} to {max_tpw:.1f} trades/week**.",
        "",
        "## Aggregate Out-of-Sample",
    ]
    for k, v in agg.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append(f"- **{k}**: {v}")

    lines.append("")
    lines.append("## Fold Highlights")
    if not folds:
        lines.append("- No valid folds were generated. Increase data range or reduce train/test window sizes.")
    else:
        for i, fold in enumerate(folds, start=1):
            p = fold["best_params"]
            t = fold["test_summary"]
            lines.append(f"- **Fold {i}** {fold['test_start']} → {fold['test_end']}")
            lines.append(
                f"  - params: mode={p['direction_mode']}, hours={p['entry_hours']}, min_prev_body_pct={p['min_prev_body_pct']}, fee={p['fee_per_trade']}"
            )
            lines.append(f"  - weekdays (Mon=0): {p['entry_weekdays']}")
            lines.append(
                f"  - OOS: trades/week={t['trades_per_week']:.3f}, total_return={t['total_return']:.6f}, sharpe={t['sharpe_approx']:.6f}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--min-trades-per-week", type=float, default=3.0)
    p.add_argument("--max-trades-per-week", type=float, default=5.0)
    p.add_argument("--hours-universe", default="0,4,8,12,16,20")
    p.add_argument("--min-body-grid", default="0.0,0.0008,0.0015")
    p.add_argument("--fee-grid", default="0.0,0.0002")
    p.add_argument("--min-train-bars", type=int, default=50)
    p.add_argument("--min-test-bars", type=int, default=20)
    p.add_argument("--output-folds", default="results/walkforward_folds.csv")
    p.add_argument("--output-report", default="results/walkforward_report.md")
    args = p.parse_args()

    rows_1h = load_1h_rows(Path(args.input))
    bars = resample_4h(rows_1h)

    hours_universe = parse_hours_csv(args.hours_universe)
    min_body_grid = [float(x.strip()) for x in args.min_body_grid.split(",") if x.strip()]
    fee_grid = [float(x.strip()) for x in args.fee_grid.split(",") if x.strip()]

    candidates = generate_candidates(hours_universe, min_body_grid, fee_grid)
    folds = walkforward(
        bars,
        train_days=args.train_days,
        test_days=args.test_days,
        min_tpw=args.min_trades_per_week,
        max_tpw=args.max_trades_per_week,
        candidates=candidates,
        min_train_bars=args.min_train_bars,
        min_test_bars=args.min_test_bars,
    )

    agg = aggregate_oos(folds, args.min_trades_per_week, args.max_trades_per_week)
    write_folds_csv(Path(args.output_folds), folds)
    write_report(Path(args.output_report), folds, agg, args.min_trades_per_week, args.max_trades_per_week)

    print("Walk-forward complete")
    for k, v in agg.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
