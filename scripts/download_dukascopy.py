"""Download hourly Dukascopy .bi5 tick files for a symbol.

Uses requests.Session per worker for HTTP keep-alive (avoids the
TLS-handshake-per-request cost that made the first version hit 15s
wall-clock per file). Default 32 workers; raise/lower via --workers.

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._dukascopy_codec import (
    DukascopyUnreachable, bi5_url, download_bi5_session, iter_hours,
    make_session, raw_path_for,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", required=True, help="UTC start YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="UTC end YYYY-MM-DD (exclusive)")
    p.add_argument("--output-dir", default="output/dukascopy")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=32,
                   help="parallel download workers (default 32)")
    p.add_argument("--unreachable-threshold", type=int, default=50,
                   help="bail if this many 403s pile up (firewall guard); "
                        "default 50 (was 5 — too aggressive for a real run "
                        "with transient errors)")
    return p.parse_args()


_thread_local = threading.local()


def _get_session(pool_size: int):
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = make_session(pool_size=pool_size)
        _thread_local.session = s
    return s


def _fetch_one(symbol: str, ts: datetime, raw_root: Path,
               retries: int, pool_size: int) -> dict:
    dst = raw_path_for(raw_root, symbol, ts)
    if dst.exists():
        body = dst.read_bytes()
        return {"ts": ts.isoformat(), "url": bi5_url(symbol, ts),
                "bytes": len(body), "status": "cached", "error": ""}
    try:
        sess = _get_session(pool_size)
        body = download_bi5_session(bi5_url(symbol, ts), dst, sess,
                                     retries=retries)
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
          f"hours={len(hours)}  workers={args.workers}  "
          f"keep-alive=session", flush=True)

    rows: list[dict] = []
    n_unreachable = 0
    bytes_total = 0
    t0 = time.time()
    last_print = t0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_fetch_one, args.symbol, ts, raw_root,
                      args.retries, args.workers): ts
            for ts in hours
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            bytes_total += r["bytes"]
            if r["status"] == "unreachable":
                n_unreachable += 1

            now = time.time()
            # progress line every 1000 files OR every 30 seconds
            if i % 1000 == 0 or i == len(hours) or (now - last_print) >= 30:
                ok = sum(1 for r in rows if r["status"] in ("ok", "cached"))
                empty = sum(1 for r in rows if r["status"] == "empty")
                elapsed = now - t0
                rate = i / max(elapsed, 0.1)
                remain_s = (len(hours) - i) / max(rate, 0.001)
                mb = bytes_total / 1024 / 1024
                print(f"  {i}/{len(hours)}  ok+cached={ok}  empty={empty}  "
                      f"unreachable={n_unreachable}  {rate:.1f} files/s  "
                      f"{mb:.1f} MB  ETA {remain_s/60:.1f} min",
                      flush=True)
                last_print = now

            if n_unreachable >= args.unreachable_threshold:
                print(f"[download] hit {n_unreachable} unreachable "
                      f"(>={args.unreachable_threshold}); aborting",
                      file=sys.stderr)
                break

    log = Path(args.output_dir) / "download_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "url", "bytes", "status", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"[download] log: {log}  total {bytes_total/1024/1024:.1f} MB  "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)

    if n_unreachable >= args.unreachable_threshold:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
