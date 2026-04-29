"""Per-trade debug charts for the CBR scalp engine (optional).

If matplotlib isn't installed, every function in this module
returns None (with a printed warning). The engine itself never
calls these; they're only invoked from `scripts/run_cbr_charts.py`
or by a notebook.

Charts produced (one PNG per trade in `out_dir`):
    candles around entry +/- N bars
    horizontal lines: prev-1H high, prev-1H low, expansion midpoint,
                      MSB level, entry, stop, target
    coloured background for the asia / execution session windows
    arrows for entry and exit
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")     # headless
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        return plt, Rectangle
    except ImportError:
        return None, None


def render_trade_chart(*,
                        trade: dict,
                        m1: pd.DataFrame,
                        out_path: Path,
                        bars_before: int = 60,
                        bars_after: int = 60,
                        levels: dict | None = None) -> Path | None:
    """Save one PNG. Returns the path on success, None if mpl missing.

    `trade` keys expected: entry_time, exit_time, direction, entry_price,
    stop_price, target_price, exit_price, exit_reason, r_result.

    `levels` (optional) keys: prev_h1_high, prev_h1_low, expansion_mid,
    msb_break_level. All as floats. Missing keys are skipped.
    """
    plt, Rectangle = _try_import_mpl()
    if plt is None:
        print("[charts] matplotlib not installed; skipping chart")
        return None

    et = pd.Timestamp(trade["entry_time"]).tz_convert("UTC") \
            if pd.Timestamp(trade["entry_time"]).tzinfo else \
            pd.Timestamp(trade["entry_time"], tz="UTC")
    xt = pd.Timestamp(trade["exit_time"]).tz_convert("UTC") \
            if pd.Timestamp(trade["exit_time"]).tzinfo else \
            pd.Timestamp(trade["exit_time"], tz="UTC")
    df = m1.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # window: bars_before before entry, bars_after after exit
    start_ts = et - pd.Timedelta(minutes=bars_before)
    end_ts = xt + pd.Timedelta(minutes=bars_after)
    win = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].reset_index(drop=True)
    if win.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    # render bars as small high-low lines + open/close ticks (no full
    # candlestick library to keep deps minimal)
    for _, r in win.iterrows():
        color = "#3fb950" if r["close"] >= r["open"] else "#f85149"
        ax.plot([r["time"], r["time"]], [r["low"], r["high"]], color=color,
                  linewidth=0.6)
        ax.plot([r["time"]], [r["open"]], marker="_", color=color, markersize=4)
        ax.plot([r["time"]], [r["close"]], marker="_", color=color, markersize=4)

    # horizontal levels
    if levels:
        for label, color, ls in [
            ("prev_h1_high", "#8b949e", "--"),
            ("prev_h1_low",  "#8b949e", "--"),
            ("expansion_mid", "#58a6ff", ":"),
            ("msb_break_level", "#d29922", "-."),
        ]:
            v = levels.get(label)
            if v is not None and pd.notna(v):
                ax.axhline(v, color=color, linestyle=ls, linewidth=0.8,
                            alpha=0.7, label=label)
    # entry / stop / target lines
    for label, value, color in [
        ("entry", trade["entry_price"], "#58a6ff"),
        ("stop", trade["stop_price"], "#f85149"),
        ("target", trade["target_price"], "#3fb950"),
    ]:
        ax.axhline(value, color=color, linewidth=1.0, label=label)

    # entry / exit markers
    arrow_color = "#58a6ff" if int(trade["direction"]) > 0 else "#f85149"
    ax.scatter([et], [trade["entry_price"]], color=arrow_color, marker="^"
                if int(trade["direction"]) > 0 else "v", s=80, zorder=6,
                label=f"entry ({trade['direction']:+d})")
    exit_color = "#3fb950" if "target" in str(trade.get("exit_reason", "")) \
                  else "#f85149"
    ax.scatter([xt], [trade["exit_price"]], color=exit_color, marker="x",
                s=80, zorder=6, label=f"exit ({trade.get('exit_reason')})")

    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("price")
    title = (f"trade {trade.get('trade_id', '?')}  "
              f"dir={int(trade['direction']):+d}  "
              f"R={trade.get('r_result', 0):.2f}  "
              f"reason={trade.get('exit_reason', '')}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(alpha=0.2, linewidth=0.4)
    fig.autofmt_xdate(rotation=30)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all_trades(trades_csv: Path | str, m1_df: pd.DataFrame,
                       output_dir: Path | str,
                       *, max_charts: int = 30) -> int:
    """Render up to `max_charts` per-trade charts. Returns count
    actually written."""
    trades = pd.read_csv(trades_csv)
    if trades.empty:
        return 0
    out = Path(output_dir)
    n_drawn = 0
    for _, row in trades.head(max_charts).iterrows():
        path = out / f"trade_{int(row['trade_id']):04d}.png"
        result = render_trade_chart(
            trade=row.to_dict(), m1=m1_df, out_path=path)
        if result is not None:
            n_drawn += 1
    return n_drawn
