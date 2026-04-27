"""Spot tests for TimeframeDataValidator."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.validator import TimeframeDataValidator, validate_h4_m15_alignment


def _make_h4(times: list[str]) -> pd.DataFrame:
    ts = pd.to_datetime(times, utc=True)
    return pd.DataFrame({
        "time": ts,
        "open": [1.0] * len(ts),
        "high": [1.5] * len(ts),
        "low": [0.5] * len(ts),
        "close": [1.2] * len(ts),
        "volume": [10.0] * len(ts),
        "spread": [0.0] * len(ts),
    })


def test_aligned_h4_passes():
    df = _make_h4(["2024-01-01 00:00:00", "2024-01-01 04:00:00", "2024-01-01 08:00:00"])
    rep = TimeframeDataValidator("h4_test", "H4").validate(df)
    assert rep.ok, rep.findings


def test_misaligned_h4_fails():
    df = _make_h4(["2024-01-01 00:30:00", "2024-01-01 04:30:00"])
    rep = TimeframeDataValidator("h4_test", "H4").validate(df)
    assert not rep.ok
    assert any(f.code == "misaligned" for f in rep.findings)


def test_duplicate_timestamps_flagged():
    df = _make_h4(["2024-01-01 00:00:00", "2024-01-01 00:00:00"])
    rep = TimeframeDataValidator("h4_test", "H4").validate(df)
    assert not rep.ok
    assert any(f.code == "duplicates" for f in rep.findings)


def test_invalid_ohlc_flagged():
    df = _make_h4(["2024-01-01 00:00:00"])
    df.loc[0, "low"] = 99.0   # low > high should fail
    rep = TimeframeDataValidator("h4_test", "H4").validate(df)
    assert not rep.ok
    assert any(f.code == "invalid_ohlc" for f in rep.findings)


def test_m15_inside_h4_buckets_passes():
    h4 = _make_h4(["2024-01-01 00:00:00", "2024-01-01 04:00:00"])
    m15_times = pd.to_datetime([
        "2024-01-01 00:00:00", "2024-01-01 00:15:00", "2024-01-01 00:30:00", "2024-01-01 00:45:00",
        "2024-01-01 04:00:00", "2024-01-01 04:15:00",
    ], utc=True)
    m15 = pd.DataFrame({
        "time": m15_times,
        "open": [1.0] * 6, "high": [1.1] * 6, "low": [0.9] * 6, "close": [1.05] * 6,
        "volume": [1.0] * 6, "spread": [0.0] * 6,
    })
    rep = validate_h4_m15_alignment(h4, m15)
    assert rep.ok, rep.findings


def test_m15_outside_h4_bucket_flagged():
    h4 = _make_h4(["2024-01-01 00:00:00"])
    m15 = pd.DataFrame({
        "time": pd.to_datetime(["2024-01-01 00:15:00", "2024-01-01 04:30:00"], utc=True),
        "open": [1.0, 1.0], "high": [1.1, 1.1], "low": [0.9, 0.9], "close": [1.0, 1.0],
        "volume": [1.0, 1.0], "spread": [0.0, 0.0],
    })
    rep = validate_h4_m15_alignment(h4, m15)
    assert not rep.ok
    assert any(f.code == "m15_outside_h4_bucket" for f in rep.findings)


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            raise
