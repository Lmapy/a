"""Download real XAUUSD OHLC bars from PUBLIC GitHub-hosted MT5 exports.

No synthetic data is generated. Sources are pinned to immutable commit
SHAs so re-running this script always returns the same bytes.

After download we record the SHA-256 of each file plus row count and
date span in results/data_manifest.json. The manifest is what the
audit PDF cites as the data provenance.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "data_manifest.json"

# Pinned commit SHAs. Update these (and re-run) when refreshing the dataset;
# never use a /HEAD/ URL here because that would silently drift.
SOURCES = [
    {
        "name": "XAUUSD_H4_long.csv",
        "repo": "142f/inv-cry",
        "commit_sha": "c9307060289ad47b7f184b206344edc7ad08c2cd",
        "path_in_repo": "data_2015_xau_btc/processed/mt5/H4/XAUUSDc.csv",
        "purpose": "long-history H4 hit-rate + walk-forward folds",
    },
    {
        "name": "XAUUSD_H4_matched.csv",
        "repo": "tiumbj/Bot_Data_Basese",
        "commit_sha": "bf8cc14e17ba8f40783144c3030c5dec3c93d812",
        "path_in_repo": "Local_LLM/dataset/tf/XAUUSD_H4.csv",
        "purpose": "matched H4 for executed-backtest holdout",
    },
    {
        "name": "XAUUSD_M15_matched.csv",
        "repo": "tiumbj/Bot_Data_Basese",
        "commit_sha": "bf8cc14e17ba8f40783144c3030c5dec3c93d812",
        "path_in_repo": "Local_LLM/dataset/tf/XAUUSD_M15.csv",
        "purpose": "matched M15 for executed-backtest entries",
    },
]


def _build_url(s: dict) -> str:
    return f"https://raw.githubusercontent.com/{s['repo']}/{s['commit_sha']}/{s['path_in_repo']}"


def download(url: str, dst: Path) -> bytes:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as r:
        body = r.read()
    dst.write_bytes(body)
    return body


TIME_COLS = ("time", "timestamp", "datetime", "Time", "Datetime")


def normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"no time column in {path.name}: cols={list(df.columns)}")
    df = df.rename(columns={tcol: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)

    if "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    elif "volume" not in df.columns:
        df["volume"] = 0.0
    if "spread" not in df.columns:
        df["spread"] = 0.0

    df = df[["time", "open", "high", "low", "close", "volume", "spread"]]
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    for s in SOURCES:
        url = _build_url(s)
        dst = DATA / s["name"]
        body = download(url, dst)
        sha256 = hashlib.sha256(body).hexdigest()
        df = normalize(dst)
        # Re-write the normalised CSV (open/high/low/close + volume + spread).
        df.to_csv(dst, index=False)
        # We hash the raw download (not the normalised file) so the manifest
        # is reproducible against the upstream source.
        entry = {
            "name": s["name"],
            "url": url,
            "repo": s["repo"],
            "commit_sha": s["commit_sha"],
            "path_in_repo": s["path_in_repo"],
            "purpose": s["purpose"],
            "raw_bytes": len(body),
            "raw_sha256": sha256,
            "rows_normalised": int(len(df)),
            "first_time_utc": str(df["time"].iloc[0]),
            "last_time_utc": str(df["time"].iloc[-1]),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        manifest_entries.append(entry)
        print(f"{s['name']:28s}  bytes={len(body):>7}  rows={len(df):>5}  "
              f"sha256={sha256[:12]}…  pinned@{s['commit_sha'][:7]}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": "All sources pinned to immutable commit SHAs. Update SOURCES "
                  "in scripts/fetch_data.py and re-run to refresh.",
        "sources": manifest_entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
