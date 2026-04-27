#!/usr/bin/env python3
"""Download hourly GC=F candles from Yahoo chart API using stdlib only."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from urllib.request import urlopen


def fetch_chart_json(symbol: str, interval: str, range_: str) -> dict:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval={interval}&range={range_}&includePrePost=false&events=div%2Csplits"
    )
    with urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_rows(payload: dict) -> list[dict]:
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]

    rows = []
    for i, ts in enumerate(timestamps):
        o = quote["open"][i]
        h = quote["high"][i]
        l = quote["low"][i]
        c = quote["close"][i]
        v = quote.get("volume", [None] * len(timestamps))[i]
        if None in (o, h, l, c):
            continue
        rows.append(
            {
                "timestamp": dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat(),
                "open": f"{o:.6f}",
                "high": f"{h:.6f}",
                "low": f"{l:.6f}",
                "close": f"{c:.6f}",
                "volume": "" if v is None else str(v),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="GC=F")
    p.add_argument("--interval", default="60m", help="Yahoo interval, e.g. 60m")
    p.add_argument("--range", dest="range_", default="730d", help="Yahoo range, e.g. 730d")
    p.add_argument("--out", default="data/gold_1h.csv")
    args = p.parse_args()

    payload = fetch_chart_json(args.symbol, args.interval, args.range_)
    if payload.get("chart", {}).get("error"):
        raise RuntimeError(payload["chart"]["error"])

    rows = parse_rows(payload)
    if not rows:
        raise RuntimeError("No rows returned.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
