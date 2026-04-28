"""Canonical research-train / validation / holdout split loader.

The pipeline must enforce that:
  * strategy generation / tuning uses `research_train` only
  * model selection uses `validation` only
  * final certification uses `holdout` only

Mixing windows is the most common way a research harness falsely
certifies a strategy. Every runner should obtain its data through
`load_splits()` (or `slice_split()`) rather than calling the loader
directly. The audit (`scripts/audit.py`) verifies that the on-disk
split boundaries are non-overlapping and contiguous, and that no
canonical runner pulls from `data.loader.load_candles` outside of
this module.

Schema in `config/data_splits.json`:

    {
      "research_train": {"start": ISO, "end": ISO},
      "validation":     {"start": ISO, "end": ISO},
      "holdout":        {"start": ISO, "end": ISO}
    }

`load_splits()` returns a `Splits` object with six DataFrames:
`train_h4`, `train_m15`, `validation_h4`, `validation_m15`,
`holdout_h4`, `holdout_m15`. Frames are time-sliced by `[start, end)`
half-open intervals so adjacent splits never share a row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.loader import load_candles

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "data_splits.json"

SPLIT_NAMES = ("research_train", "validation", "holdout")


@dataclass
class Split:
    name: str
    start: pd.Timestamp     # inclusive, tz-aware UTC
    end: pd.Timestamp       # exclusive, tz-aware UTC

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time" not in df.columns:
            raise ValueError("expected a 'time' column on the input frame")
        m = (df["time"] >= self.start) & (df["time"] < self.end)
        return df.loc[m].reset_index(drop=True)


@dataclass
class Splits:
    train: Split
    validation: Split
    holdout: Split
    train_h4: pd.DataFrame
    train_m15: pd.DataFrame
    validation_h4: pd.DataFrame
    validation_m15: pd.DataFrame
    holdout_h4: pd.DataFrame
    holdout_m15: pd.DataFrame


def _ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def load_split_config(path: Path | None = None) -> dict[str, Split]:
    """Read the split config and return ordered Split objects.

    Raises if the three windows are missing, overlap, are out of order,
    or have non-positive duration.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    cfg = json.loads(cfg_path.read_text())
    out: dict[str, Split] = {}
    for n in SPLIT_NAMES:
        if n not in cfg:
            raise ValueError(f"data_splits.json missing required split '{n}'")
        s = _ts(cfg[n]["start"])
        e = _ts(cfg[n]["end"])
        if e <= s:
            raise ValueError(f"split '{n}': end <= start ({s} .. {e})")
        out[n] = Split(name=n, start=s, end=e)
    # ordering + non-overlap
    ordered = [out[n] for n in SPLIT_NAMES]
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt.start < prev.end:
            raise ValueError(
                f"splits overlap: {prev.name} ends {prev.end} but "
                f"{nxt.name} starts {nxt.start}")
    return out


def load_splits(symbol: str = "XAUUSD",
                config_path: Path | None = None) -> Splits:
    """Load Dukascopy H4 + M15 and pre-slice into the three splits.

    Use this from every runner that needs train/validation/holdout
    data. Do NOT call `load_candles()` from a runner -- it bypasses
    the split contract.
    """
    cfg = load_split_config(config_path)
    train_cfg = cfg["research_train"]
    val_cfg = cfg["validation"]
    ho_cfg = cfg["holdout"]

    h4 = load_candles(symbol=symbol, timeframe="H4")
    m15 = load_candles(symbol=symbol, timeframe="M15")

    return Splits(
        train=train_cfg,
        validation=val_cfg,
        holdout=ho_cfg,
        train_h4=train_cfg.slice(h4),
        train_m15=train_cfg.slice(m15),
        validation_h4=val_cfg.slice(h4),
        validation_m15=val_cfg.slice(m15),
        holdout_h4=ho_cfg.slice(h4),
        holdout_m15=ho_cfg.slice(m15),
    )


def assert_no_holdout_leak(df: pd.DataFrame, splits: Splits, *,
                           where: str) -> None:
    """Fail loudly if `df` contains any row whose timestamp falls inside
    the holdout window. Call from research/validation runners."""
    if "time" not in df.columns or len(df) == 0:
        return
    leaks = df[(df["time"] >= splits.holdout.start) &
               (df["time"] <  splits.holdout.end)]
    if len(leaks):
        raise AssertionError(
            f"holdout leakage detected in {where}: {len(leaks)} rows "
            f"in [{splits.holdout.start}, {splits.holdout.end})")


def assert_only_in_split(df: pd.DataFrame, split: Split, *, where: str) -> None:
    """Fail loudly if `df` contains any row outside `split` boundaries."""
    if "time" not in df.columns or len(df) == 0:
        return
    outside = df[(df["time"] < split.start) | (df["time"] >= split.end)]
    if len(outside):
        raise AssertionError(
            f"{where}: {len(outside)} rows outside split '{split.name}' "
            f"[{split.start}, {split.end})")


def coverage_summary(splits: Splits) -> dict[str, dict]:
    """Diagnostic: data coverage per split. Useful for the audit when
    config declares a wider window than data actually covers."""
    out: dict[str, dict] = {}
    for name, frame in [
        ("research_train", splits.train_h4),
        ("validation", splits.validation_h4),
        ("holdout", splits.holdout_h4),
    ]:
        cfg = getattr(splits, "train" if name == "research_train" else name)
        if len(frame):
            actual_start = frame["time"].iloc[0]
            actual_end = frame["time"].iloc[-1]
        else:
            actual_start = actual_end = None
        out[name] = {
            "config_start": str(cfg.start),
            "config_end": str(cfg.end),
            "actual_first_bar": str(actual_start) if actual_start is not None else None,
            "actual_last_bar":  str(actual_end)   if actual_end   is not None else None,
            "h4_rows": int(len(frame)),
        }
    return out
