from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import pandas as pd


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def to_unix_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def from_unix_ms(v: int) -> datetime:
    return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)


def iter_time_chunks(start: datetime, end: datetime, chunk_days: int) -> Iterator[tuple[datetime, datetime]]:
    cur = start
    delta = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt


def ensure_utc_index(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True)
    return out
