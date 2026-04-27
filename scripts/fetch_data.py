"""Download real XAUUSD OHLC data from public GitHub-hosted MT5 exports.

No synthetic data is generated. We pull two real datasets:

  data/XAUUSD_H4_long.csv
      Long-history 4-hour bars used for the *hit-rate diagnostic* of the
      "next bar continues prior bar's direction" hypothesis.
      Source: github.com/142f/inv-cry  (MT5 export, 2018-06-28 -> present).

  data/XAUUSD_H4_matched.csv  +  data/XAUUSD_M15_matched.csv
      Matched 4-hour and 15-minute bars from the same broker, used by the
      executed backtest where entries are placed on the M15 timeframe.
      Source: github.com/tiumbj/Bot_Data_Basese  (MT5 export, 2026-01-30 -> 2026-04-01).
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "XAUUSD_H4_long.csv":
        "https://raw.githubusercontent.com/142f/inv-cry/HEAD/"
        "data_2015_xau_btc/processed/mt5/H4/XAUUSDc.csv",
    "XAUUSD_H4_matched.csv":
        "https://raw.githubusercontent.com/tiumbj/Bot_Data_Basese/HEAD/"
        "Local_LLM/dataset/tf/XAUUSD_H4.csv",
    "XAUUSD_M15_matched.csv":
        "https://raw.githubusercontent.com/tiumbj/Bot_Data_Basese/HEAD/"
        "Local_LLM/dataset/tf/XAUUSD_M15.csv",
}


def download(url: str, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as r:
        body = r.read()
    dst.write_bytes(body)
    return len(body)


TIME_COLS = ("time", "timestamp", "datetime", "Time", "Datetime")


def normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"no time column in {path.name}: cols={list(df.columns)}")
    df = df.rename(columns={tcol: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)

    keep = ["time", "open", "high", "low", "close"]
    if "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    elif "volume" in df.columns:
        pass
    else:
        df["volume"] = 0.0
    if "spread" not in df.columns:
        df["spread"] = 0.0

    df = df[keep + ["volume", "spread"]]
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dst = DATA / name
        n = download(url, dst)
        df = normalize(dst)
        df.to_csv(dst, index=False)
        print(f"{name:28s}  bytes={n:>7}  rows={len(df):>5}  "
              f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
