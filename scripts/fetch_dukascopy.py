"""Dukascopy XAUUSD official fetcher.

Downloads hourly .bi5 LZMA-compressed tick files from
   https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5

Each record is 20 bytes big-endian:
    int32   ms since hour start
    int32   ask price (raw int; XAUUSD point size = 0.001)
    int32   bid price
    float32 ask volume
    float32 bid volume

Pipeline:
    1. download tick files for [start, end)
    2. decode bi5 -> per-tick rows
    3. resample to M1/M3/M5/M15/M30/H1/H4/D1 OHLC (open=first, high=max,
       low=min, close=last, volume=sum, spread_mean, spread_max,
       tick_count)
    4. write CSVs into data/dukascopy/candles/XAUUSD/<TF>/year=<YYYY>.csv
    5. write data/dukascopy/manifests/XAUUSD_manifest.json with policy
       + per-year SHA256 + bar counts

Fails LOUDLY with `DukascopyUnreachable` when the upstream host is
blocked, so the audit downstream correctly reports the gap rather
than silently substituting non-official data.

POLICY: Dukascopy is the SINGLE official data source. No mixing with
other brokers. dataset_source = "dukascopy" is written into every
output candle row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DUKA = ROOT / "data" / "dukascopy"
RAW = DUKA / "raw"
CANDLES = DUKA / "candles" / "XAUUSD"
MANIFEST = DUKA / "manifests" / "XAUUSD_manifest.json"

# XAUUSD: point size 0.001 (Dukascopy convention; verified against
# their published spec page). Volume is in lots.
POINT_SIZE = 0.001

TIMEFRAMES_MIN = {
    "M1": 1, "M3": 3, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


class DukascopyUnreachable(RuntimeError):
    """Raised when the upstream Dukascopy CDN is not reachable. The
    rest of the pipeline must NOT fall back to a different broker."""


# ---------- low-level downloader ----------

def _bi5_url(instrument: str, ts_utc: datetime) -> str:
    # Note: month is 0-indexed in the Dukascopy URL convention.
    return (f"https://datafeed.dukascopy.com/datafeed/{instrument}/"
            f"{ts_utc.year:04d}/{ts_utc.month - 1:02d}/{ts_utc.day:02d}/"
            f"{ts_utc.hour:02d}h_ticks.bi5")


def _download_bi5(url: str, dst: Path, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "a-research-bot/1.0"})
    try:
        with urlopen(req, timeout=timeout) as r:
            body = r.read()
    except HTTPError as e:
        if e.code in (403, 404):
            # 403 from sandbox firewall, 404 = no ticks for that hour
            # (markets closed). We surface 403 as DukascopyUnreachable
            # because the URL itself is correct.
            if e.code == 403:
                raise DukascopyUnreachable(
                    f"Dukascopy returned 403 for {url}. From this "
                    "environment the firewall blocks "
                    "datafeed.dukascopy.com. Run scripts/fetch_dukascopy.py "
                    "outside the sandbox, or drop pre-fetched .bi5 / OHLC "
                    "CSVs under data/dukascopy/raw/ or "
                    "data/dukascopy/candles/XAUUSD/<TF>/.") from e
            return b""    # 404 = empty hour (weekend, holiday) -- treat as no ticks
    except URLError as e:
        raise DukascopyUnreachable(f"Cannot reach Dukascopy: {e}") from e
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(body)
    return body


# ---------- bi5 decoder ----------

_RECORD_FMT = ">iiiff"
_RECORD_SIZE = 20


def _decode_bi5(body: bytes, hour_utc: datetime, point: float) -> pd.DataFrame:
    if not body:
        return pd.DataFrame(columns=["time", "bid", "ask", "ask_vol", "bid_vol"])
    raw = lzma.decompress(body)
    n = len(raw) // _RECORD_SIZE
    if n == 0:
        return pd.DataFrame(columns=["time", "bid", "ask", "ask_vol", "bid_vol"])
    times: list[pd.Timestamp] = []
    bids = np.empty(n, dtype=float)
    asks = np.empty(n, dtype=float)
    ask_vol = np.empty(n, dtype=float)
    bid_vol = np.empty(n, dtype=float)
    for i in range(n):
        ms, ask_raw, bid_raw, av, bv = struct.unpack_from(
            _RECORD_FMT, raw, i * _RECORD_SIZE)
        times.append(hour_utc + timedelta(milliseconds=ms))
        asks[i] = ask_raw * point
        bids[i] = bid_raw * point
        ask_vol[i] = av
        bid_vol[i] = bv
    df = pd.DataFrame({
        "time": pd.to_datetime(times, utc=True),
        "bid": bids, "ask": asks,
        "ask_vol": ask_vol, "bid_vol": bid_vol,
    })
    return df


# ---------- ticks -> OHLC ----------

def _resample(ticks: pd.DataFrame, freq: str) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame(columns=[
            "time", "open", "high", "low", "close", "volume",
            "spread_mean", "spread_max", "tick_count", "dataset_source",
        ])
    df = ticks.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["volume"] = df["ask_vol"] + df["bid_vol"]
    df = df.set_index("time")
    out = df.resample(freq, closed="left", label="left").agg(
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
        volume=("volume", "sum"),
        spread_mean=("spread", "mean"),
        spread_max=("spread", "max"),
        tick_count=("mid", "count"),
    ).dropna(subset=["open"]).reset_index()
    out["dataset_source"] = "dukascopy"
    return out


# ---------- main fetch ----------

def fetch_range(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    cur = start
    rows: list[pd.DataFrame] = []
    while cur < end:
        rel = (f"{cur.year:04d}/{cur.month-1:02d}/{cur.day:02d}/"
               f"{cur.hour:02d}h_ticks.bi5")
        local = RAW / instrument / rel
        if local.exists():
            body = local.read_bytes()
        else:
            url = _bi5_url(instrument, cur)
            body = _download_bi5(url, local)
        if body:
            rows.append(_decode_bi5(body, cur, POINT_SIZE))
        cur += timedelta(hours=1)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("time")


def write_candles(ticks: pd.DataFrame) -> dict[str, dict]:
    """Resample to all 8 timeframes; write per-year CSVs; return summary."""
    if ticks.empty:
        return {}
    ticks = ticks.sort_values("time").reset_index(drop=True)
    summary: dict[str, dict] = {}
    for tf, minutes in TIMEFRAMES_MIN.items():
        # pandas: '1min', '3min', '5min', '15min', '30min', '60min', '240min', '1440min'
        freq = f"{minutes}min"
        bars = _resample(ticks, freq)
        if bars.empty:
            continue
        # write per-year for manageable file sizes
        bars["year"] = bars["time"].dt.year
        out_dir = CANDLES / tf
        out_dir.mkdir(parents=True, exist_ok=True)
        per_year = {}
        for year, g in bars.groupby("year"):
            cols = ["time", "open", "high", "low", "close", "volume",
                    "spread_mean", "spread_max", "tick_count", "dataset_source"]
            path = out_dir / f"year={year}.csv"
            g[cols].to_csv(path, index=False)
            per_year[str(year)] = {
                "rows": int(len(g)),
                "first": str(g["time"].iloc[0]),
                "last": str(g["time"].iloc[-1]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        summary[tf] = {"total_rows": int(len(bars)), "by_year": per_year}
    return summary


def write_manifest(summary: dict, instrument: str) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "official_source": "dukascopy",
        "symbol": instrument,
        "timeframes": list(TIMEFRAMES_MIN.keys()),
        "old_sources_deprecated": True,
        "generated_from_tick_data": True,
        "point_size": POINT_SIZE,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": ("Dukascopy is the single official data source for the "
                   "active research, validation, holdout, prop simulation, "
                   "and reporting pipeline. No mixing of broker sources. "
                   "Old broker data lives under data/_deprecated_/ and is "
                   "marked dataset_source != dukascopy; the audit and "
                   "certifier reject it for active use."),
        "summary": summary,
    }
    MANIFEST.write_text(json.dumps(doc, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--start", required=True, help="UTC start date YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="UTC end date   YYYY-MM-DD (exclusive)")
    args = ap.parse_args()
    s = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    print(f"fetching {args.symbol} ticks  {s} -> {e}", flush=True)
    try:
        ticks = fetch_range(args.symbol, s, e)
    except DukascopyUnreachable as ex:
        print(f"\nDUKASCOPY UNREACHABLE FROM THIS ENVIRONMENT.\n  {ex}\n",
              file=sys.stderr)
        print("Pipeline expects same-source Dukascopy data — refusing to "
              "substitute. Re-run from an unrestricted network or drop "
              "pre-fetched .bi5 hourly files under "
              "data/dukascopy/raw/<INSTRUMENT>/<YYYY>/<MM-1>/<DD>/ "
              "(or pre-resampled OHLC under "
              "data/dukascopy/candles/XAUUSD/<TF>/year=YYYY.csv).",
              file=sys.stderr)
        return 2
    print(f"  ticks fetched: {len(ticks):,}", flush=True)
    summary = write_candles(ticks)
    write_manifest(summary, args.symbol)
    print(f"  wrote candles for: {sorted(summary.keys())}", flush=True)
    print(f"  manifest: {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
