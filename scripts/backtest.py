"""4h candle continuation backtest on XAUUSD.

Hypothesis (user-stated):
    "On every 4h candle open, price probably continues in the same direction
     as the previous 4h candle."

Two stages of analysis:

  STAGE 1 - Hit-rate diagnostic (long-history H4):
      Empirically measure P( color_t == color_{t-1} ) on 8.6k H4 bars
      from 2018-06-28 to 2026-04-20.  Tells us whether the hypothesis
      is supported by the data at all -- before any execution model.

  STAGE 2 - Executed backtest with M15 entries (matched H4 + M15):
      For each H4 bar t, look at the previous H4 bar (t-1) for direction.
      Then act on the 15-minute timeframe within H4 bar t:

        m15_open       -> enter at the open of H4 bar t's first M15 sub-bar
                          (same price as H4 open; sanity baseline).
        m15_confirm    -> wait for the first M15 sub-bar in t whose body
                          confirms the predicted direction; enter at its close.
                          If no confirmation arrives, no trade.
        m15_atr_stop   -> m15_open entry, stop placed 1 * M15-ATR(14) the wrong
                          way; exit at H4 close or stop hit (whichever first).

      Position direction is +1 or -1, fixed at notional = 1.
      Exit price is always the H4 close, unless stopped out earlier.
      Spread cost is taken from M15 'spread' column (full round-trip).

Outputs (results/):
    hit_rate.csv         -- Stage 1 conditional probabilities.
    summary.csv          -- Stage 2 per-strategy metrics.
    trades.csv           -- Stage 2 per-trade ledger.
    equity.csv / .png    -- Stage 2 equity curves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results"

H4_LONG = DATA / "XAUUSD_H4_long.csv"
H4 = DATA / "XAUUSD_H4_matched.csv"
M15 = DATA / "XAUUSD_M15_matched.csv"

H4_BARS_PER_YEAR = 1560     # ~6 bars/day * ~5 days/wk * 52 wk
H4_HOURS = 4


# ---------- helpers ----------

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


# ---------- Stage 1: hit-rate diagnostic ----------

def hit_rate(h4: pd.DataFrame) -> pd.DataFrame:
    color = np.sign(h4["close"] - h4["open"]).astype(int)
    pair = pd.DataFrame({"prev": color.shift(1), "cur": color}).dropna()
    pair = pair[(pair["prev"] != 0) & (pair["cur"] != 0)]

    p_same = (pair["prev"] == pair["cur"]).mean()
    n = len(pair)

    # Conditional P( cur == +1 | prev color )
    cond_up_after_up = ((pair["prev"] == 1) & (pair["cur"] == 1)).sum() / max((pair["prev"] == 1).sum(), 1)
    cond_dn_after_dn = ((pair["prev"] == -1) & (pair["cur"] == -1)).sum() / max((pair["prev"] == -1).sum(), 1)

    # Wald 95% CI on p_same
    se = math.sqrt(p_same * (1 - p_same) / n) if n > 0 else 0.0
    lo, hi = p_same - 1.96 * se, p_same + 1.96 * se

    return pd.DataFrame([
        {"metric": "bars_tested", "value": n},
        {"metric": "P(same direction)", "value": round(float(p_same), 4)},
        {"metric": "P(same) 95% CI low", "value": round(float(lo), 4)},
        {"metric": "P(same) 95% CI high", "value": round(float(hi), 4)},
        {"metric": "P(up | prev up)", "value": round(float(cond_up_after_up), 4)},
        {"metric": "P(down | prev down)", "value": round(float(cond_dn_after_dn), 4)},
        {"metric": "edge over 50% (pp)", "value": round(float(p_same - 0.5) * 100, 3)},
        {"metric": "data span", "value": f"{h4['time'].iloc[0].date()} -> {h4['time'].iloc[-1].date()}"},
    ])


# ---------- Stage 2: executed backtest ----------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry: float
    exit: float
    cost: float
    pnl: float
    ret: float


def assign_h4_bucket(m15: pd.DataFrame) -> pd.Series:
    """Map each M15 bar to the H4 bar it belongs to (floor to 4h)."""
    return m15["time"].dt.floor(f"{H4_HOURS}h")


def build_h4_with_signal(h4: pd.DataFrame) -> pd.DataFrame:
    """Add prior-bar direction as the entry signal."""
    h4 = h4.copy()
    h4["color"] = np.sign(h4["close"] - h4["open"]).astype(int)
    h4["sig"] = h4["color"].shift(1)        # +1 buy, -1 sell, 0 skip
    return h4


def simulate(
    h4: pd.DataFrame,
    m15: pd.DataFrame,
    mode: str,
    use_costs: bool = True,
    atr_n: int = 14,
) -> tuple[list[Trade], dict]:
    assert mode in {"m15_open", "m15_confirm", "m15_atr_stop"}

    h4 = build_h4_with_signal(h4)
    m15 = m15.copy()
    m15["bucket"] = assign_h4_bucket(m15)
    m15["m15_atr"] = atr(m15, atr_n)
    m15_by_bucket = {k: g for k, g in m15.groupby("bucket")}

    trades: list[Trade] = []
    skipped_no_subbars = 0
    skipped_no_confirm = 0
    skipped_no_signal = 0

    for _, row in h4.iterrows():
        sig = row["sig"]
        if pd.isna(sig) or sig == 0:
            skipped_no_signal += 1
            continue

        bucket = row["time"]
        sub = m15_by_bucket.get(bucket)
        if sub is None or len(sub) == 0:
            skipped_no_subbars += 1
            continue
        sub = sub.sort_values("time").reset_index(drop=True)

        if mode == "m15_open":
            entry_row = sub.iloc[0]
            entry_price = float(entry_row["open"])
            entry_time = entry_row["time"]
            stop_price = None
        elif mode == "m15_confirm":
            confirms = sub[np.sign(sub["close"] - sub["open"]) == sig]
            if confirms.empty:
                skipped_no_confirm += 1
                continue
            entry_row = confirms.iloc[0]
            entry_price = float(entry_row["close"])
            entry_time = entry_row["time"]
            stop_price = None
        else:  # m15_atr_stop
            entry_row = sub.iloc[0]
            entry_price = float(entry_row["open"])
            entry_time = entry_row["time"]
            a = entry_row.get("m15_atr")
            if pd.isna(a):
                stop_price = None
            else:
                stop_price = entry_price - sig * float(a)

        # Exit logic: scan M15 bars at/after entry; check stop hits, else H4 close.
        exit_time = sub.iloc[-1]["time"]
        exit_price = float(row["close"])
        # Track which sub-bar the exit fires on so the exit-leg spread comes
        # from the actual exit bar (default = bucket-final bar for time exits).
        exit_row = sub.iloc[-1]
        if stop_price is not None:
            future = sub[sub["time"] >= entry_time]
            for _, b in future.iterrows():
                hi, lo = float(b["high"]), float(b["low"])
                if sig > 0 and lo <= stop_price:
                    exit_time = b["time"]
                    exit_price = stop_price
                    exit_row = b
                    break
                if sig < 0 and hi >= stop_price:
                    exit_time = b["time"]
                    exit_price = stop_price
                    exit_row = b
                    break

        # Cost: round-trip spread (price units) summed at entry & exit.
        # Use the spread from the actual EXIT bar -- when a stop fires intra-
        # bucket the bucket-final spread is the wrong one to charge.
        cost_price = 0.0
        if use_costs:
            sp = float(entry_row.get("spread", 0.0)) + float(exit_row.get("spread", 0.0))
            point = 0.001  # XAUUSDc point size from broker spec; spread is in points
            cost_price = sp * point

        gross = sig * (exit_price - entry_price)
        pnl = gross - cost_price
        ret = pnl / entry_price

        trades.append(Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=int(sig),
            entry=entry_price,
            exit=exit_price,
            cost=cost_price,
            pnl=float(pnl),
            ret=float(ret),
        ))

    diag = {
        "skipped_no_signal": skipped_no_signal,
        "skipped_no_subbars": skipped_no_subbars,
        "skipped_no_confirm": skipped_no_confirm,
    }
    return trades, diag


def trades_to_metrics(name: str, trades: list[Trade]) -> dict:
    if not trades:
        return {"strategy": name, "trades": 0, "wins": 0, "win_rate": 0.0,
                "avg_ret_bp": 0.0, "total_return": 0.0, "sharpe_ann": 0.0,
                "max_drawdown": 0.0, "avg_hold_min": 0.0}

    rets = pd.Series([t.ret for t in trades])
    eq = (1 + rets).cumprod()
    wins = int((rets > 0).sum())
    win_rate = wins / len(rets)
    avg_bp = float(rets.mean() * 10_000)
    total = float(eq.iloc[-1] - 1.0)
    sd = float(rets.std(ddof=1))
    sharpe = (rets.mean() / sd) * math.sqrt(H4_BARS_PER_YEAR) if sd > 0 else 0.0
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min())
    hold = float(np.mean([
        (t.exit_time - t.entry_time).total_seconds() / 60.0 for t in trades
    ]))
    return {
        "strategy": name,
        "trades": len(trades),
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "avg_ret_bp": round(avg_bp, 3),
        "total_return": round(total, 4),
        "sharpe_ann": round(float(sharpe), 3),
        "max_drawdown": round(dd, 4),
        "avg_hold_min": round(hold, 1),
    }


# ---------- main ----------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    h4_long = load(H4_LONG)
    h4 = load(H4)
    m15 = load(M15)

    # Stage 1
    hr = hit_rate(h4_long)
    hr.to_csv(OUT / "hit_rate.csv", index=False)
    print("=== STAGE 1 hit-rate diagnostic (long-history H4) ===")
    print(hr.to_string(index=False))
    print()

    # Stage 2
    print("=== STAGE 2 executed backtest (matched M15 entries) ===")
    print(f"H4 bars: {len(h4)}  M15 bars: {len(m15)}  "
          f"window: {h4['time'].iloc[0]} -> {h4['time'].iloc[-1]}")

    summaries = []
    eq_curves = {}
    trade_rows = []
    for mode in ("m15_open", "m15_confirm", "m15_atr_stop"):
        trades, diag = simulate(h4, m15, mode=mode, use_costs=True)
        m = trades_to_metrics(mode, trades)
        m.update({f"diag_{k}": v for k, v in diag.items()})
        summaries.append(m)

        if trades:
            df = pd.DataFrame([{
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "dir": t.direction, "entry": t.entry, "exit": t.exit,
                "cost": t.cost, "pnl": t.pnl, "ret": t.ret, "strategy": mode,
            } for t in trades])
            trade_rows.append(df)
            eq_curves[mode] = (1 + df.set_index("exit_time")["ret"]).cumprod()

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "summary.csv", index=False)
    print(summary_df.to_string(index=False))

    if trade_rows:
        pd.concat(trade_rows, ignore_index=True).to_csv(OUT / "trades.csv", index=False)

    if eq_curves:
        eq_df = pd.concat(eq_curves, axis=1, sort=True).ffill()
        eq_df.to_csv(OUT / "equity.csv")

        fig, ax = plt.subplots(figsize=(11, 6))
        for col in eq_df.columns:
            ax.plot(eq_df.index, eq_df[col], label=col, linewidth=1.2)
        ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title("XAUUSD H4-continuation w/ M15 entries — equity curve (1.0 = flat)")
        ax.set_ylabel("equity multiple")
        ax.set_xlabel("time")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "equity.png", dpi=130)

    print(f"\nWrote: {OUT}/hit_rate.csv, summary.csv, trades.csv, equity.csv, equity.png")


if __name__ == "__main__":
    main()
