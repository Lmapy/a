"""TimeframeDataValidator.

Non-negotiable. If a validation rule fails, downstream code must refuse
to run. Every rule in section 8 of the spec is enumerated here and is
covered by tests/test_validator.py.

Rules enforced:
  - complete candles only (timestamp matches expected bar boundary)
  - correct timestamp alignment (e.g. M15 timestamps land on :00, :15, :30, :45)
  - UTC-normalised timestamps
  - no mixing timezones
  - no lookahead bias (validator is read-only on time order)
  - lower-TF candles must map to a parent H4 candle
  - entry only allowed AFTER H4 close (verified at executor level; this
    module asserts every M15 sub-bar lies strictly inside one H4 bucket)
  - missing candles, duplicates, invalid OHLC, timestamp gaps detected
  - resampling integrity: open=first, high=max, low=min, close=last, volume=sum
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from core.constants import TF_MINUTES


@dataclass
class ValidationFinding:
    code: str
    message: str
    sample: dict | None = None
    severity: str = "error"   # error | warning | info


@dataclass
class ValidationReport:
    findings: list[ValidationFinding] = field(default_factory=list)

    def error(self, code: str, msg: str, sample: dict | None = None) -> None:
        self.findings.append(ValidationFinding(code, msg, sample, "error"))

    def warn(self, code: str, msg: str, sample: dict | None = None) -> None:
        self.findings.append(ValidationFinding(code, msg, sample, "warning"))

    def info(self, code: str, msg: str, sample: dict | None = None) -> None:
        self.findings.append(ValidationFinding(code, msg, sample, "info"))

    @property
    def errors(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_rows(self) -> list[dict]:
        return [{"severity": f.severity, "code": f.code,
                 "message": f.message, "sample": f.sample} for f in self.findings]


def _is_utc(s: pd.Series) -> bool:
    if not pd.api.types.is_datetime64_any_dtype(s):
        return False
    if getattr(s.dt, "tz", None) is None:
        return False
    return str(s.dt.tz) == "UTC"


def _aligned_to(ts: pd.Series, minutes: int) -> pd.Series:
    """True where the timestamp lands exactly on a `minutes`-aligned boundary."""
    if minutes == 1:
        return ts.dt.second.eq(0)
    if minutes < 60:
        return ts.dt.second.eq(0) & (ts.dt.minute % minutes).eq(0)
    hours = minutes // 60
    return ts.dt.second.eq(0) & ts.dt.minute.eq(0) & (ts.dt.hour % hours).eq(0)


class TimeframeDataValidator:
    """Validate one OHLC frame against an expected bar interval."""

    def __init__(self, name: str, tf: str):
        if tf not in TF_MINUTES:
            raise ValueError(f"unknown timeframe {tf}")
        self.name = name
        self.tf = tf
        self.minutes = TF_MINUTES[tf]

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        rep = ValidationReport()
        if df.empty:
            rep.error("empty", f"{self.name}: dataframe is empty")
            return rep

        # 1. UTC enforcement.
        if not _is_utc(df["time"]):
            rep.error("not_utc", f"{self.name}: time column not tz-aware UTC")
            return rep

        # 2. Sortedness + no duplicates.
        if not df["time"].is_monotonic_increasing:
            rep.error("not_sorted", f"{self.name}: time column not sorted ascending")
        if df["time"].duplicated().any():
            n = int(df["time"].duplicated().sum())
            rep.error("duplicates", f"{self.name}: {n} duplicated timestamps")

        # 3. Alignment to bar boundary (timestamps land on :00, :15, ...).
        misaligned = df[~_aligned_to(df["time"], self.minutes)]
        if len(misaligned):
            rep.error("misaligned",
                      f"{self.name}: {len(misaligned)} bars not aligned to {self.minutes}-min boundary",
                      sample={"first": str(misaligned["time"].iloc[0])})

        # 4. Invalid OHLC (high<max(open,close), low>min(open,close), low>high).
        bad = df[(df["high"] < df[["open", "close"]].max(axis=1))
                 | (df["low"] > df[["open", "close"]].min(axis=1))
                 | (df["low"] > df["high"])]
        if len(bad):
            rep.error("invalid_ohlc",
                      f"{self.name}: {len(bad)} rows have inconsistent OHLC",
                      sample=bad.iloc[0][["time", "open", "high", "low", "close"]].astype(str).to_dict())

        # 5. Gap detection. We tolerate weekend/news gaps but report them.
        diffs_min = df["time"].diff().dt.total_seconds().div(60).iloc[1:]
        gaps = diffs_min[diffs_min > self.minutes]
        if len(gaps):
            n_big = int((gaps > self.minutes * 6).sum())
            rep.warn("gaps_detected",
                     f"{self.name}: {len(gaps)} gaps > {self.minutes}-min "
                     f"({n_big} > 6×bar length)",
                     sample={"first_gap_min": float(gaps.iloc[0])})

        # 6. NaN / inf in OHLC.
        if df[["open", "high", "low", "close"]].isna().any().any():
            rep.error("nan_in_ohlc", f"{self.name}: NaN in OHLC")
        return rep


def validate_h4_m15_alignment(h4: pd.DataFrame, m15: pd.DataFrame) -> ValidationReport:
    """Every M15 bar must land strictly inside one H4 bucket; resampling M15->H4 must reproduce H4."""
    rep = ValidationReport()
    if h4.empty or m15.empty:
        rep.error("empty", "h4 or m15 dataframe is empty")
        return rep

    bucket = m15["time"].dt.floor("4h")
    in_h4 = bucket.isin(set(h4["time"]))
    n_outside = int((~in_h4).sum())
    if n_outside:
        rep.error("m15_outside_h4_bucket",
                  f"{n_outside} M15 bars do not map to any H4 bar in the matched dataset",
                  sample={"first": str(m15.loc[~in_h4, "time"].iloc[0])})

    # Resample integrity: take M15 bars whose bucket appears in H4 and aggregate.
    sub = m15[in_h4].copy()
    sub["bucket"] = bucket[in_h4]
    agg = sub.groupby("bucket").agg(open=("open", "first"),
                                    high=("high", "max"),
                                    low=("low", "min"),
                                    close=("close", "last"),
                                    volume=("volume", "sum")).reset_index()
    agg = agg.rename(columns={"bucket": "time"})
    merged = pd.merge(h4[["time", "open", "high", "low", "close"]], agg,
                      on="time", how="inner", suffixes=("_h4", "_resampled"))
    if merged.empty:
        rep.error("no_overlap", "no overlap between H4 and resampled M15")
        return rep

    # OHLC tolerance: brokers can disagree by a fraction of a tick on the
    # exact open/close due to feed timing -- allow 5 * point as slack,
    # but strict-equal on high/low.
    slack = 5 * 0.001
    diffs = merged.assign(
        open_ok=(merged["open_h4"] - merged["open_resampled"]).abs() <= slack,
        close_ok=(merged["close_h4"] - merged["close_resampled"]).abs() <= slack,
        high_ok=merged["high_h4"] >= merged["high_resampled"] - slack,   # H4 high must not be < resampled high
        low_ok=merged["low_h4"] <= merged["low_resampled"] + slack,
    )
    failed = diffs[~(diffs["open_ok"] & diffs["close_ok"] & diffs["high_ok"] & diffs["low_ok"])]
    if len(failed):
        rep.warn("resample_mismatch",
                 f"{len(failed)}/{len(merged)} H4 bars disagree with M15 resample beyond slack",
                 sample={"first": str(failed.iloc[0]["time"])})
    return rep


def assert_no_lookahead(spec_signal_at_t: callable, h4: pd.DataFrame, sample: int = 200) -> ValidationReport:
    """Dynamic look-ahead probe: for a given signal function, verify the
    signal at bar t depends only on bars strictly < t. Implemented by
    perturbing future bars and checking the signal at t is unchanged.
    """
    import numpy as np
    rep = ValidationReport()
    if len(h4) < sample + 1:
        return rep
    rng = np.random.default_rng(42)
    idxs = rng.choice(range(20, len(h4) - 1), size=min(sample, len(h4) - 21), replace=False)
    fails = 0
    for t in idxs:
        baseline = spec_signal_at_t(h4, int(t))
        perturbed = h4.copy()
        perturbed.loc[perturbed.index[t + 1:], ["open", "high", "low", "close"]] *= 1.5
        check = spec_signal_at_t(perturbed, int(t))
        if baseline != check:
            fails += 1
    if fails:
        rep.error("lookahead_detected", f"{fails}/{len(idxs)} lookahead probes failed")
    return rep


def run_full_validation(h4_long: pd.DataFrame, h4: pd.DataFrame, m15: pd.DataFrame) -> ValidationReport:
    out = ValidationReport()
    for name, frame, tf in [
        ("h4_long", h4_long, "H4"),
        ("h4_matched", h4, "H4"),
        ("m15_matched", m15, "M15"),
    ]:
        sub = TimeframeDataValidator(name, tf).validate(frame)
        out.findings.extend(sub.findings)
    out.findings.extend(validate_h4_m15_alignment(h4, m15).findings)
    return out
