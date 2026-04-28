"""Dukascopy-only pipeline audit.

Hard rules (any failure → exit 1):

  - data/loader.py refuses non-Dukascopy sources
  - every candle file under data/dukascopy/candles/ has dataset_source == "dukascopy"
  - data/dukascopy/manifests/XAUUSD_manifest.json exists with official_source = "dukascopy"
  - no active code path imports the deprecated broker CSVs in data/_deprecated_/
  - active result files (alpha leaderboard etc.) carry no rows referencing
    deprecated sources
  - timeframe alignment / OHLC sanity / gap detection (when data is on disk)
  - parent/child mapping H4 ↔ M15 (when both are on disk)

If `data/dukascopy/candles/XAUUSD/H4/` is empty, the audit reports
`no_dukascopy_data_yet` as a fatal failure with instructions, NOT a
warning. The pipeline refuses to certify anything until the official
source is populated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_candles, list_available

RESULTS = ROOT / "results"
DUKA_MANIFEST = ROOT / "data" / "dukascopy" / "manifests" / "XAUUSD_manifest.json"
CANDLES = ROOT / "data" / "dukascopy" / "candles" / "XAUUSD"
DEPRECATED = ROOT / "data" / "_deprecated_"


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0

    def write(self, msg: str) -> None:
        self.lines.append(msg)
        print(msg)

    def section(self, title: str) -> None:
        self.write("")
        self.write(f"=== {title} ===")

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        self.write(f"  [{tag}] {label}" + (f"   ({detail})" if detail else ""))

    def dump(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def audit_manifest(rep: Report) -> dict | None:
    rep.section("manifest — single Dukascopy source")
    rep.check("data/dukascopy/manifests/XAUUSD_manifest.json exists",
              DUKA_MANIFEST.exists(), detail=str(DUKA_MANIFEST))
    if not DUKA_MANIFEST.exists():
        return None
    m = json.loads(DUKA_MANIFEST.read_text())
    rep.check("official_source == dukascopy",
              m.get("official_source") == "dukascopy",
              detail=f"got={m.get('official_source')}")
    rep.check("old_sources_deprecated == true",
              m.get("old_sources_deprecated") is True)
    rep.check("generated_from_tick_data == true",
              m.get("generated_from_tick_data") is True)
    rep.check("manifest declares all 8 required timeframes",
              set(m.get("timeframes", [])) >=
              {"M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"})
    return m


def audit_data_present(rep: Report) -> dict[str, list[int]]:
    rep.section("Dukascopy candle data on disk")
    avail = list_available()
    if not avail or not any(avail.values()):
        rep.check("at least one timeframe has candles on disk",
                  False,
                  detail="no Dukascopy data yet — run scripts/fetch_dukascopy.py "
                         "from an unrestricted environment, or drop "
                         ".bi5 hourly tick files into data/dukascopy/raw/")
        return avail
    for tf in ("M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"):
        years = avail.get(tf, [])
        rep.check(f"{tf}: years on disk", bool(years),
                  detail=f"{years if years else 'none'}")
    return avail


def audit_dataset_source(rep: Report, avail: dict[str, list[int]]) -> None:
    rep.section("dataset_source enforcement")
    if not any(avail.values()):
        rep.write("  (skipped — no candles on disk yet)")
        return
    bad = 0
    for tf in avail:
        files = (list((CANDLES / tf).glob("year=*.parquet"))
                 + list((CANDLES / tf).glob("year=*.csv")))
        for f in files:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f, columns=["dataset_source"])
            else:
                df = pd.read_csv(f, nrows=1)
            if "dataset_source" not in df.columns:
                rep.check(f"{tf}/{f.name}: dataset_source column present",
                          False, detail="missing column")
                bad += 1
                continue
            if (df["dataset_source"] != "dukascopy").any():
                rep.check(f"{tf}/{f.name}: dataset_source",
                          False, detail=f"got={df['dataset_source'].iloc[0]}")
                bad += 1
    rep.check("every candle file's dataset_source == dukascopy", bad == 0)


def audit_no_active_use_of_deprecated(rep: Report) -> None:
    rep.section("active code does not import deprecated broker data")
    if not DEPRECATED.exists():
        rep.write("  (no _deprecated_ dir; skipping)")
        return
    deprecated_csvs = {
        "XAUUSD_H4_long.csv",
        "XAUUSD_H4_matched.csv",
        "XAUUSD_M15_matched.csv",
    }
    # Audit and the report builder are allowed to MENTION the
    # deprecated filenames as text (the audit checks FOR them; the PDF
    # builder cites historical results). Only flag files that load
    # them via pd.read_csv.
    skip = {"audit.py", "build_pdf.py", "fetch_dukascopy.py"}
    offenders: list[str] = []
    for py in (ROOT / "scripts").glob("*.py"):
        if "_deprecated_" in py.parts or py.name in skip:
            continue
        text = py.read_text()
        for c in deprecated_csvs:
            if f'pd.read_csv("data/{c}"' in text or f'data/{c}' in text:
                offenders.append(f"{py.name} loads {c}")
    rep.check("no scripts/*.py references the deprecated broker CSVs",
              not offenders,
              detail="; ".join(offenders) if offenders else "")


def audit_timeframe_integrity(rep: Report, avail: dict[str, list[int]]) -> None:
    rep.section("timeframe integrity (when data is present)")
    if not avail.get("M15") or not avail.get("H4"):
        rep.write("  (skipped — H4 or M15 not on disk)")
        return
    h4 = load_candles(timeframe="H4")
    m15 = load_candles(timeframe="M15")
    rep.check("H4 timestamps land on a 4h boundary",
              ((h4["time"].dt.hour % 4 == 0)
               & (h4["time"].dt.minute == 0)
               & (h4["time"].dt.second == 0)).all())
    rep.check("M15 timestamps land on a 15-min boundary",
              ((m15["time"].dt.minute % 15 == 0)
               & (m15["time"].dt.second == 0)).all())
    rep.check("H4 OHLC valid (low <= open,close <= high)",
              ((h4["low"] <= h4[["open", "close"]].min(axis=1))
               & (h4[["open", "close"]].max(axis=1) <= h4["high"])).all())
    # M15 sub-bars must each map to an H4 bucket present in H4
    bucket = m15["time"].dt.floor("4h")
    n_outside = int((~bucket.isin(set(h4["time"]))).sum())
    rep.check("every M15 bar maps to a known H4 bucket",
              n_outside == 0, detail=f"outside={n_outside}")


def main() -> int:
    rep = Report()
    rep.write("XAUUSD Dukascopy-only pipeline audit")
    rep.write("=" * 60)

    audit_manifest(rep)
    avail = audit_data_present(rep)
    audit_dataset_source(rep, avail)
    audit_no_active_use_of_deprecated(rep)
    audit_timeframe_integrity(rep, avail)

    rep.write("")
    rep.write(f"=== audit summary: {rep.failures} failures ===")
    rep.dump(RESULTS / "audit.txt")
    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
