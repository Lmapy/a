"""Download hourly Dukascopy .bi5 tick files for a symbol.

Writes raw files into:

    <output_dir>/raw/<SYMBOL>/<YYYY>/<MM-1:02d>/<DD:02d>/<HH:02d>h_ticks.bi5

A 0-byte file means the upstream returned 404 (closed-market hour); we
cache it so re-runs don't re-fetch.

Writes <output_dir>/download_log.csv with one row per attempted hour.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._dukascopy_codec import (
    DukascopyUnreachable, bi5_url, download_bi5, iter_hours, raw_path_for,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", required=True, help="UTC start YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="UTC end YYYY-MM-DD (exclusive)")
    p.add_argument("--output-dir", default="output/dukascopy")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def _fetch_one(symbol: str, ts: datetime, raw_root: Path, retries: int) -> dict:
    dst = raw_path_for(raw_root, symbol, ts)
    if dst.exists():
        # already cached (incl. 0-byte 404 cache)
        body = dst.read_bytes()
        return {"ts": ts.isoformat(), "url": bi5_url(symbol, ts),
                "bytes": len(body), "status": "cached", "error": ""}
    try:
        body = download_bi5(bi5_url(symbol, ts), dst, retries=retries)
    except DukascopyUnreachable as e:
        return {"ts": ts.isoformat(), "url": bi5_url(symbol, ts),
                "bytes": 0, "status": "unreachable", "error": str(e)}
    status = "empty" if not body else "ok"
    return {"ts": ts.isoformat(), "url": bi5_url(symbol, ts),
            "bytes": len(body), "status": status, "error": ""}


def main() -> int:
    args = parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    raw_root = Path(args.output_dir) / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    hours = list(iter_hours(start, end))
    print(f"[download] {args.symbol}  {start.date()} -> {end.date()}  "
          f"hours={len(hours)}  workers={args.workers}", flush=True)

    rows: list[dict] = []
    n_unreachable = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one, args.symbol, ts, raw_root, args.retries): ts
                   for ts in hours}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            if r["status"] == "unreachable":
                n_unreachable += 1
            if i % 500 == 0 or i == len(hours):
                ok = sum(1 for r in rows if r["status"] in ("ok", "cached"))
                empty = sum(1 for r in rows if r["status"] == "empty")
                print(f"  {i}/{len(hours)}  ok+cached={ok}  empty={empty}  "
                      f"unreachable={n_unreachable}", flush=True)
            if n_unreachable >= 5:
                # bail early — running 6 years through a firewalled CDN
                # would be 50k more 403s
                print("[download] too many unreachable hours; aborting",
                      file=sys.stderr)
                break

    log = Path(args.output_dir) / "download_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "url", "bytes", "status", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"[download] log: {log}", flush=True)

    if n_unreachable:
        return 2  # signal upstream-blocked
    return 0


if __name__ == "__main__":
    sys.exit(main())
