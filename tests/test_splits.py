"""Split-loader tests.

These tests fail loudly if:
  - splits overlap or are out of order in config
  - end <= start
  - any row in train/validation accidentally falls inside the holdout
  - the slicer leaks across boundaries
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.splits import (
    Split,
    Splits,
    load_split_config,
    load_splits,
    assert_no_holdout_leak,
    assert_only_in_split,
    coverage_summary,
)


# ---------- config-level tests (no data files needed) ----------

def _write_cfg(d: dict) -> Path:
    f = NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, f)
    f.flush()
    f.close()
    return Path(f.name)


def test_load_config_happy_path():
    cfg_path = _write_cfg({
        "research_train": {"start": "2020-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"},
        "validation":     {"start": "2022-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
        "holdout":        {"start": "2024-01-01T00:00:00Z", "end": "2026-05-01T00:00:00Z"},
    })
    cfg = load_split_config(cfg_path)
    assert cfg["research_train"].start.tzinfo is not None
    assert cfg["research_train"].end == cfg["validation"].start
    assert cfg["validation"].end == cfg["holdout"].start


def test_overlapping_splits_rejected():
    """Validation starts before train ends -> must raise."""
    cfg_path = _write_cfg({
        "research_train": {"start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
        "validation":     {"start": "2022-06-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},  # overlaps train
        "holdout":        {"start": "2024-01-01T00:00:00Z", "end": "2026-05-01T00:00:00Z"},
    })
    with pytest.raises(ValueError, match="overlap"):
        load_split_config(cfg_path)


def test_inverted_split_rejected():
    """end <= start -> must raise."""
    cfg_path = _write_cfg({
        "research_train": {"start": "2020-01-01T00:00:00Z", "end": "2019-01-01T00:00:00Z"},  # inverted
        "validation":     {"start": "2022-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
        "holdout":        {"start": "2024-01-01T00:00:00Z", "end": "2026-05-01T00:00:00Z"},
    })
    with pytest.raises(ValueError, match="end <= start"):
        load_split_config(cfg_path)


def test_missing_split_rejected():
    cfg_path = _write_cfg({
        "research_train": {"start": "2020-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"},
        "validation":     {"start": "2022-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
        # holdout missing
    })
    with pytest.raises(ValueError, match="missing required split 'holdout'"):
        load_split_config(cfg_path)


def test_split_slice_is_half_open():
    """The end of train should NOT be in train; it should be in validation."""
    s_train = Split(name="research_train",
                    start=pd.Timestamp("2020-01-01", tz="UTC"),
                    end=pd.Timestamp("2022-01-01", tz="UTC"))
    s_val = Split(name="validation",
                  start=pd.Timestamp("2022-01-01", tz="UTC"),
                  end=pd.Timestamp("2024-01-01", tz="UTC"))
    df = pd.DataFrame({"time": pd.to_datetime([
        "2021-12-31 23:00", "2022-01-01 00:00", "2022-01-01 01:00",
    ], utc=True)})
    train = s_train.slice(df)
    val = s_val.slice(df)
    assert len(train) == 1
    assert len(val) == 2
    # the boundary timestamp 2022-01-01 00:00 must appear in exactly one slice
    boundary = pd.Timestamp("2022-01-01", tz="UTC")
    assert (train["time"] == boundary).sum() == 0
    assert (val["time"] == boundary).sum() == 1


# ---------- data-level tests (require dukascopy candles on disk) ----------

def _have_candles() -> bool:
    return (ROOT / "data" / "dukascopy" / "candles" / "XAUUSD" / "H4").exists()


@pytest.mark.skipif(not _have_candles(),
                    reason="Dukascopy candles not on disk; run scripts/pull_sidecar.py")
def test_load_splits_no_overlap_in_data():
    splits = load_splits()
    # Pairwise: no row in one split appears in another (by timestamp).
    for a_name, a in [("train", splits.train_h4),
                      ("validation", splits.validation_h4),
                      ("holdout", splits.holdout_h4)]:
        for b_name, b in [("train", splits.train_h4),
                          ("validation", splits.validation_h4),
                          ("holdout", splits.holdout_h4)]:
            if a_name >= b_name:
                continue
            shared = set(a["time"]) & set(b["time"])
            assert not shared, (
                f"{a_name} and {b_name} share {len(shared)} timestamps")


@pytest.mark.skipif(not _have_candles(),
                    reason="Dukascopy candles not on disk")
def test_holdout_leak_detector():
    splits = load_splits()
    # train data should not trip the leak guard
    assert_no_holdout_leak(splits.train_h4, splits, where="test_self")
    # but a frame that includes holdout rows MUST trip
    full = pd.concat([splits.train_h4, splits.holdout_h4], ignore_index=True)
    with pytest.raises(AssertionError, match="holdout leakage"):
        assert_no_holdout_leak(full, splits, where="leaky_frame")


@pytest.mark.skipif(not _have_candles(),
                    reason="Dukascopy candles not on disk")
def test_only_in_split_guard():
    splits = load_splits()
    # train data is all inside train -- must pass
    assert_only_in_split(splits.train_h4, splits.train, where="test_self")
    # but train data must NOT pass for the validation split
    with pytest.raises(AssertionError, match="outside split 'validation'"):
        assert_only_in_split(splits.train_h4, splits.validation, where="cross_check")


@pytest.mark.skipif(not _have_candles(),
                    reason="Dukascopy candles not on disk")
def test_coverage_summary_present():
    splits = load_splits()
    cov = coverage_summary(splits)
    for k in ("research_train", "validation", "holdout"):
        assert k in cov
        assert cov[k]["h4_rows"] >= 0
        assert "config_start" in cov[k]


# ---------- in-process driver ----------
if __name__ == "__main__":
    failures = 0
    fns = [
        test_load_config_happy_path,
        test_overlapping_splits_rejected,
        test_inverted_split_rejected,
        test_missing_split_rejected,
        test_split_slice_is_half_open,
    ]
    if _have_candles():
        fns.extend([
            test_load_splits_no_overlap_in_data,
            test_holdout_leak_detector,
            test_only_in_split_guard,
            test_coverage_summary_present,
        ])
    else:
        print("  SKIP  data-level tests (run scripts/pull_sidecar.py first)")
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    if failures:
        raise SystemExit(1)
