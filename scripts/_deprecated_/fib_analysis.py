"""Fib-level reaction analysis on XAUUSD H4.

For every H4 bar i with non-zero prior-bar color, we project standard
Fibonacci retracement levels onto the previous bar's range:

    long  (after green):  level price = prev_high - f * (prev_high - prev_low)
    short (after red):    level price = prev_low  + f * (prev_high - prev_low)

For each level f in {0.236, 0.382, 0.5, 0.618, 0.786, 1.0} we then ask:

    touched_i(f)        = did the current H4 bar's range reach the level?
    reaction_i(f)       = touched AND H4 closed in the trade direction?
    ret_from_touch_i(f) = signed return from the level price to H4 close,
                          if touched. (sign = trade direction)

Outputs:

    results/fib_levels.csv          aggregate per level (long history,
                                    matched window, and "trend candle"
                                    subset).
    results/fib_deepest.csv         distribution of deepest level
                                    touched per H4 bar.

The diagnostic is independent of any execution model -- it answers
"where on the previous candle does price tend to react?" without
assuming we trade every level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data"
OUT = ROOT / "results"

LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)


def per_bar_table(h4: pd.DataFrame) -> pd.DataFrame:
    """One row per H4 bar with the level prices and per-level touch/return."""
    df = h4.copy()
    df["color"] = np.sign(df["close"] - df["open"]).astype(int)
    df["sig"] = df["color"].shift(1).fillna(0).astype(int)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_open"] = df["open"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    df["range"] = df["prev_high"] - df["prev_low"]
    df = df[df["sig"] != 0].copy()
    df = df[df["range"] > 0].copy()

    rows = []
    for f in LEVELS:
        long_lvl = df["prev_high"] - f * df["range"]
        short_lvl = df["prev_low"] + f * df["range"]
        level_price = np.where(df["sig"] > 0, long_lvl, short_lvl)
        # touched: long if low <= level; short if high >= level
        touched_long = (df["low"].values <= level_price) & (df["sig"].values > 0)
        touched_short = (df["high"].values >= level_price) & (df["sig"].values < 0)
        touched = touched_long | touched_short
        # signed return from the level price to H4 close (in trade direction)
        ret = np.where(
            touched,
            df["sig"].values * (df["close"].values - level_price) / level_price,
            np.nan,
        )
        # reaction: touched AND H4 closed in trade direction
        reaction = touched & ((df["sig"].values * (df["close"].values - df["open"].values)) > 0)
        for j in range(len(df)):
            rows.append({
                "time": df["time"].iloc[j],
                "sig": int(df["sig"].iloc[j]),
                "level": f,
                "level_price": float(level_price[j]),
                "touched": bool(touched[j]),
                "reaction": bool(reaction[j]),
                "ret_from_touch": float(ret[j]) if not np.isnan(ret[j]) else np.nan,
            })
    return pd.DataFrame(rows)


def aggregate(per_bar: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for f, g in per_bar.groupby("level"):
        n = len(g)
        n_touched = int(g["touched"].sum())
        n_react = int(g["reaction"].sum())
        rets = g.loc[g["touched"], "ret_from_touch"]
        rows.append({
            "dataset": label,
            "level": f,
            "trades_evaluated": n,
            "touched": n_touched,
            "touch_rate": round(n_touched / n, 4) if n else 0.0,
            "reaction_count": n_react,
            "reaction_rate": round(n_react / n, 4) if n else 0.0,
            "reaction_given_touch": round(n_react / n_touched, 4) if n_touched else 0.0,
            "mean_ret_from_touch_bp": round(float(rets.mean() * 10_000), 2) if len(rets) else 0.0,
            "median_ret_from_touch_bp": round(float(rets.median() * 10_000), 2) if len(rets) else 0.0,
            "win_rate_from_touch": round(float((rets > 0).mean()), 4) if len(rets) else 0.0,
        })
    return pd.DataFrame(rows)


def deepest_touched(per_bar: pd.DataFrame, label: str) -> pd.DataFrame:
    """For each H4 bar, find the deepest level (largest f) that was touched."""
    pivot = per_bar.pivot(index="time", columns="level", values="touched").fillna(False)
    deepest = []
    for ts, row in pivot.iterrows():
        levels_touched = [c for c in pivot.columns if bool(row[c])]
        deepest.append(max(levels_touched) if levels_touched else np.nan)
    s = pd.Series(deepest, index=pivot.index, name="deepest")
    counts = s.value_counts(dropna=False).sort_index()
    out = counts.rename_axis("deepest_level").reset_index(name="count")
    out["share"] = (out["count"] / out["count"].sum()).round(4)
    out.insert(0, "dataset", label)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h4_long = load(DATA / "XAUUSD_H4_long.csv")
    h4_match = load(DATA / "XAUUSD_H4_matched.csv")

    pb_long = per_bar_table(h4_long)
    pb_match = per_bar_table(h4_match)

    # "trend" subset on long history: prev candle body / ATR(14) >= 0.5.
    h4l = h4_long.copy()
    pc = h4l["close"].shift(1)
    tr = pd.concat([
        h4l["high"] - h4l["low"],
        (h4l["high"] - pc).abs(),
        (h4l["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    body_ratio_prev = ((h4l["close"] - h4l["open"]).abs() / atr14).shift(1)
    trend_times = h4l.loc[body_ratio_prev >= 0.5, "time"]
    pb_trend = pb_long[pb_long["time"].isin(set(trend_times))].copy()

    agg = pd.concat([
        aggregate(pb_long, "h4_long_2018_2026"),
        aggregate(pb_trend, "h4_long_trend_only_body>=0.5atr"),
        aggregate(pb_match, "h4_matched_2026"),
    ], ignore_index=True)
    agg.to_csv(OUT / "fib_levels.csv", index=False)

    deepest = pd.concat([
        deepest_touched(pb_long, "h4_long_2018_2026"),
        deepest_touched(pb_trend, "h4_long_trend_only_body>=0.5atr"),
        deepest_touched(pb_match, "h4_matched_2026"),
    ], ignore_index=True)
    deepest.to_csv(OUT / "fib_deepest.csv", index=False)

    print("=== Fib level reaction (long history 2018-2026) ===")
    show = agg[agg["dataset"] == "h4_long_2018_2026"].sort_values("level")
    print(show[["level", "touch_rate", "reaction_rate", "reaction_given_touch",
                "mean_ret_from_touch_bp", "win_rate_from_touch"]].to_string(index=False))

    print("\n=== Deepest level touched (long history) ===")
    print(deepest[deepest["dataset"] == "h4_long_2018_2026"].to_string(index=False))

    print(f"\nWrote: {OUT}/fib_levels.csv, fib_deepest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
