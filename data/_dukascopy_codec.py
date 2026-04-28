"""Shared Dukascopy codec.

Dukascopy publishes hourly LZMA-compressed `.bi5` tick files at:

    https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5

(month is zero-indexed: January = 00, December = 11)

Each tick record is 20 bytes, big-endian:

    int32   ms since hour start
    int32   ask price (raw int; XAUUSD scale = 1000 -> price = raw * 0.001)
    int32   bid price
    float32 ask volume
    float32 bid volume

This module is the single source of truth for that format; both
scripts/fetch_dukascopy.py (the local CLI) and the
scripts/{download,build,validate,run}_dukascopy_*.py CI scripts share
it.
"""
from __future__ import annotations

import json
import lzma
import struct
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_CONFIG = ROOT / "config" / "dukascopy_symbols.json"

TIMEFRAMES_MIN = {
    "M1": 1, "M3": 3, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}

CANDLE_COLS = [
    "time", "open", "high", "low", "close",
    "volume", "spread_mean", "spread_max", "tick_count", "dataset_source",
]

_RECORD_FMT = ">iiiff"
_RECORD_SIZE = 20

# Numpy structured dtype that lets us decode an entire .bi5 file in one
# vectorised call (~50× faster than the per-record struct.unpack loop).
_TICK_DTYPE = np.dtype([
    ("ms",  ">i4"),
    ("ask", ">i4"),
    ("bid", ">i4"),
    ("av",  ">f4"),
    ("bv",  ">f4"),
])


class DukascopyUnreachable(RuntimeError):
    """Upstream CDN is unreachable. Pipeline must NOT fall back to a
    different broker — the Dukascopy-only rule forbids substitution."""


# ---------- symbol metadata ----------

@dataclass
class SymbolSpec:
    symbol: str
    price_scale: int
    expected_price_min: float
    expected_price_max: float

    @property
    def point_size(self) -> float:
        return 1.0 / self.price_scale

    def sanity_warning(self, sample_mid: float) -> str | None:
        if not np.isfinite(sample_mid):
            return f"sample mid is non-finite: {sample_mid}"
        if not (self.expected_price_min <= sample_mid <= self.expected_price_max):
            return (f"sample mid {sample_mid:.4f} for {self.symbol} is "
                    f"outside the expected gold-price range "
                    f"[{self.expected_price_min}, {self.expected_price_max}] — "
                    "is PRICE_SCALE correct?")
        return None


def load_symbol_spec(symbol: str) -> SymbolSpec:
    if SYMBOLS_CONFIG.exists():
        cfg = json.loads(SYMBOLS_CONFIG.read_text())
        s = cfg.get(symbol)
        if s is not None:
            return SymbolSpec(
                symbol=symbol,
                price_scale=int(s["price_scale"]),
                expected_price_min=float(s["expected_price_min"]),
                expected_price_max=float(s["expected_price_max"]),
            )
    # sensible defaults for XAUUSD
    if symbol == "XAUUSD":
        return SymbolSpec(symbol="XAUUSD", price_scale=1000,
                          expected_price_min=200.0, expected_price_max=10000.0)
    raise ValueError(f"unknown symbol {symbol!r}; add it to {SYMBOLS_CONFIG}")


# ---------- URL + raw-path helpers ----------

def bi5_url(instrument: str, ts_utc: datetime) -> str:
    """Dukascopy URL. Note month is zero-indexed."""
    return (f"https://datafeed.dukascopy.com/datafeed/{instrument}/"
            f"{ts_utc.year:04d}/{ts_utc.month - 1:02d}/{ts_utc.day:02d}/"
            f"{ts_utc.hour:02d}h_ticks.bi5")


def raw_path_for(raw_dir: Path, instrument: str, ts_utc: datetime) -> Path:
    return (raw_dir / instrument / f"{ts_utc.year:04d}"
            / f"{ts_utc.month - 1:02d}" / f"{ts_utc.day:02d}"
            / f"{ts_utc.hour:02d}h_ticks.bi5")


# ---------- download ----------

def download_bi5(url: str, dst: Path, *, timeout: int = 30, retries: int = 3,
                 backoff: float = 2.0) -> bytes:
    """Returns body bytes. Empty bytes mean a 404 (closed market hour).

    Single-request urllib path. Kept for the local
    scripts/fetch_dukascopy.py CLI; the CI downloader uses
    download_bi5_session() which reuses TLS connections via
    requests.Session for an order-of-magnitude speedup."""
    err: Exception | None = None
    for attempt in range(retries):
        req = Request(url, headers={"User-Agent": "dukascopy-pipeline/1.0"})
        try:
            with urlopen(req, timeout=timeout) as r:
                body = r.read()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(body)
            return body
        except HTTPError as e:
            if e.code == 404:
                # Empty hour (weekend / market closed). Cache an empty file
                # so we don't retry next run.
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"")
                return b""
            if e.code == 403:
                raise DukascopyUnreachable(
                    f"403 on {url} — upstream blocked by your network. "
                    "Run from an environment that can reach "
                    "datafeed.dukascopy.com.") from e
            err = e
        except URLError as e:
            err = e
        if attempt + 1 < retries:
            time.sleep(backoff * (2 ** attempt))
    raise DukascopyUnreachable(f"failed to fetch {url} after {retries} attempts: {err}") from err


# ---------- fast session-pooled downloader (used by the CI runner) ----------

def make_session(pool_size: int):
    """Create a requests.Session with connection pooling sized for
    `pool_size` parallel workers. Imported lazily so the local
    fetch_dukascopy.py path keeps working with stdlib only."""
    import requests
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    s.headers.update({"User-Agent": "dukascopy-pipeline/1.0"})
    a = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size,
                    max_retries=0)
    s.mount("https://", a)
    s.mount("http://", a)
    return s


def download_bi5_session(url: str, dst: Path, session, *, timeout: int = 30,
                         retries: int = 5, backoff: float = 1.0) -> bytes:
    """Like download_bi5 but reuses TLS connections through a requests
    Session. Each parallel worker should hold its own session (sessions
    are not fully thread-safe under heavy contention).

    403 is treated as RETRYABLE (Dukascopy rate-limits aggressive
    concurrency by returning 403, not 429). After `retries` retries of
    403/429/5xx the request is finally classified as unreachable.
    """
    err: Exception | None = None
    last_status: int | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, stream=False)
        except Exception as e:    # ConnectionError, Timeout, etc.
            err = e
            if attempt + 1 < retries:
                # exponential backoff with jitter -- spreads workers
                # apart when Dukascopy rate-limits a burst
                import random
                time.sleep(backoff * (2 ** attempt) * (1 + 0.3 * random.random()))
            continue
        last_status = r.status_code
        if r.status_code == 200:
            body = r.content
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(body)
            return body
        if r.status_code == 404:
            # closed-market hour; cache as empty so we don't refetch
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            return b""
        if r.status_code in (403, 429) or 500 <= r.status_code < 600:
            # rate-limited / server error -> retry with backoff
            err = RuntimeError(f"HTTP {r.status_code} on {url}")
            if attempt + 1 < retries:
                import random
                time.sleep(backoff * (2 ** attempt) * (1 + 0.3 * random.random()))
            continue
        # any other status -> retry once
        err = RuntimeError(f"HTTP {r.status_code} on {url}")
        if attempt + 1 < retries:
            time.sleep(backoff)
    # all retries exhausted
    raise DukascopyUnreachable(
        f"failed to fetch {url} after {retries} attempts "
        f"(last_status={last_status}): {err}") from err


# ---------- decode ----------

def decode_bi5(body: bytes, hour_utc: datetime, price_scale: int) -> pd.DataFrame:
    """One hour of bi5 bytes -> per-tick DataFrame. Empty -> empty frame.

    Vectorised via numpy.frombuffer on a structured dtype, then
    np.byteswap to handle big-endian on little-endian hardware. ~50×
    faster than the per-record struct.unpack loop the first cut used.
    """
    empty_cols = ["time", "bid", "ask", "mid", "spread",
                  "bid_volume", "ask_volume", "volume"]
    if not body:
        return pd.DataFrame(columns=empty_cols)
    raw = lzma.decompress(body)
    n = len(raw) // _RECORD_SIZE
    if n == 0:
        return pd.DataFrame(columns=empty_cols)
    arr = np.frombuffer(raw[: n * _RECORD_SIZE], dtype=_TICK_DTYPE).copy()
    # frombuffer keeps the big-endian dtype; .astype to native-endian
    # numeric types is fastest with explicit cast.
    point = 1.0 / price_scale
    asks = arr["ask"].astype(np.float64) * point
    bids = arr["bid"].astype(np.float64) * point
    av = arr["av"].astype(np.float64)
    bv = arr["bv"].astype(np.float64)
    ms = arr["ms"].astype(np.int64)
    base_ns = np.int64(hour_utc.timestamp() * 1_000_000_000)
    times = pd.to_datetime(base_ns + ms * 1_000_000, utc=True)
    return pd.DataFrame({
        "time": times,
        "bid": bids, "ask": asks,
        "mid": (bids + asks) / 2.0,
        "spread": asks - bids,
        "bid_volume": bv, "ask_volume": av,
        "volume": bv + av,
    })


# ---------- ticks -> OHLC ----------

def resample_ticks(ticks: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame(columns=CANDLE_COLS)
    df = ticks.set_index("time")
    freq = f"{freq_minutes}min"
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
    return out[CANDLE_COLS]


# ---------- iteration ----------

def iter_hours(start: datetime, end: datetime):
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    cur = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        yield cur
        cur += timedelta(hours=1)


def chunk_ranges(start: datetime, end: datetime, chunk_by: str = "year"):
    """Yield (chunk_start, chunk_end) covering [start, end). chunk_by ∈
    {year, quarter, month}."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    cur = start
    while cur < end:
        if chunk_by == "year":
            nxt = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        elif chunk_by == "quarter":
            q = (cur.month - 1) // 3
            month = q * 3 + 1
            nxt_m = month + 3
            nxt_y = cur.year
            if nxt_m > 12:
                nxt_m -= 12
                nxt_y += 1
            nxt = datetime(nxt_y, nxt_m, 1, tzinfo=timezone.utc)
        elif chunk_by == "month":
            nxt_y = cur.year + (1 if cur.month == 12 else 0)
            nxt_m = 1 if cur.month == 12 else cur.month + 1
            nxt = datetime(nxt_y, nxt_m, 1, tzinfo=timezone.utc)
        else:
            raise ValueError(f"unknown chunk_by={chunk_by}")
        yield cur, min(nxt, end)
        cur = nxt
