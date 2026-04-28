"""End-to-end Dukascopy pipeline for the GitHub Actions workflow:

  1. resolve [start, end) from --start-date/--end-date or --years-back
  2. download hourly .bi5 ticks (with retries; cached)
  3. decode + resample into M1/M3/M5/M15/M30/H1/H4/D1 parquet
  4. validate every timeframe
  5. write manifest.json under <output-dir>/<SYMBOL>/
  6. cleanup raw/ unless --keep-raw

Final stdout block prints the summary the workflow's logs surface.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._dukascopy_codec import TIMEFRAMES_MIN, load_symbol_spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--years-back", type=int, default=6)
    p.add_argument("--start-date", default=None, help="UTC YYYY-MM-DD")
    p.add_argument("--end-date",   default=None, help="UTC YYYY-MM-DD (exclusive)")
    p.add_argument("--output-dir", default="output/dukascopy")
    p.add_argument("--write-csv", action="store_true")
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--chunk-by", choices=["year", "quarter", "month"],
                   default="year")
    p.add_argument("--workers", type=int, default=32,
                   help="parallel download workers (default 32; uses keep-alive sessions)")
    p.add_argument("--ticks-only", action="store_true",
                   help="download raw .bi5 ticks and stop. The artifact will "
                        "be the raw/ tree; user runs build_dukascopy_candles.py "
                        "locally to decode + resample. Useful when CI is the "
                        "only host with internet access to Dukascopy and the "
                        "user wants to iterate on candle logic locally.")
    return p.parse_args()


def _resolve_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    if args.start_date and args.end_date:
        s = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
        e = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    else:
        e = today
        s = today - timedelta(days=int(args.years_back) * 365)
    if e <= s:
        raise SystemExit(f"end ({e}) must be > start ({s})")
    return s, e


def _run(cmd: list[str]) -> int:
    print(f"\n+ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    args = parse_args()
    spec = load_symbol_spec(args.symbol)   # validates symbol known
    start, end = _resolve_range(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== Dukascopy pipeline for {args.symbol} ==")
    print(f"   range: {start.date()}  ->  {end.date()}")
    print(f"   chunk_by={args.chunk_by}  workers={args.workers}  "
          f"write_csv={args.write_csv}  keep_raw={args.keep_raw}")

    # 1. download
    rc = _run([sys.executable, "scripts/download_dukascopy.py",
               "--symbol", args.symbol,
               "--start", start.date().isoformat(),
               "--end",   end.date().isoformat(),
               "--output-dir", str(out_dir),
               "--workers", str(args.workers)])
    if rc != 0:
        print("[pipeline] download step failed", file=sys.stderr)
        return rc

    if args.ticks_only:
        # Skip build + validate. The artifact is the raw .bi5 tree;
        # the user runs build_dukascopy_candles.py locally.
        print("[pipeline] --ticks-only: skipping build + validate. "
              "Artifact will be the raw/ tree.", flush=True)
        sym_dir = out_dir / args.symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "official_source": "dukascopy",
            "symbol": args.symbol,
            "price_scale": spec.price_scale,
            "expected_price_range": [spec.expected_price_min, spec.expected_price_max],
            "range_start_utc": start.isoformat(),
            "range_end_utc": end.isoformat(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "ticks_only",
            "policy": ("Dukascopy is the single official data source. Run "
                       "scripts/build_dukascopy_candles.py locally on the "
                       "raw/ tree to produce M1..D1 OHLC parquet."),
            "next_step": (f"python3 scripts/build_dukascopy_candles.py "
                          f"--symbol {args.symbol} --start {start.date()} "
                          f"--end {end.date()} --input-dir <unzipped raw/> "
                          f"--output-dir <where you want candles>"),
        }
        (sym_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print()
        print("=" * 60)
        print(f"Dukascopy {args.symbol} pipeline complete (ticks_only mode)")
        print(f"  Symbol:      {args.symbol}")
        print(f"  Start:       {start.date()}")
        print(f"  End:         {end.date()}")
        print(f"  Raw tree:    {out_dir / 'raw'}")
        print(f"  Manifest:    {sym_dir / 'manifest.json'}")
        print(f"  Next step:   run scripts/build_dukascopy_candles.py locally")
        print("=" * 60)
        return 0

    # 2. build
    rc = _run([sys.executable, "scripts/build_dukascopy_candles.py",
               "--symbol", args.symbol,
               "--start", start.date().isoformat(),
               "--end",   end.date().isoformat(),
               "--input-dir",  str(out_dir),
               "--output-dir", str(out_dir),
               "--chunk-by", args.chunk_by,
               *(["--write-csv"] if args.write_csv else [])])
    if rc != 0:
        print("[pipeline] build step failed", file=sys.stderr)
        return rc

    # 3. validate
    rc_v = _run([sys.executable, "scripts/validate_dukascopy.py",
                 "--symbol", args.symbol,
                 "--input-dir", str(out_dir)])
    # rc_v != 0 means validation found hard failures, but we still want
    # the manifest written so the artifact captures the failure detail.

    # 4. manifest
    sym_dir = out_dir / args.symbol
    rows_per_tf: dict[str, int] = {}
    for tf in TIMEFRAMES_MIN:
        pq = sym_dir / tf / f"{args.symbol}_{tf}.parquet"
        if pq.exists():
            try:
                import pyarrow.parquet as pq_lib
                rows_per_tf[tf] = int(pq_lib.ParquetFile(pq).metadata.num_rows)
            except Exception:
                import pandas as pd
                rows_per_tf[tf] = int(len(pd.read_parquet(pq)))
        else:
            rows_per_tf[tf] = 0
    val_path = sym_dir / "validation_report.json"
    val = json.loads(val_path.read_text()) if val_path.exists() else {}
    manifest = {
        "official_source": "dukascopy",
        "symbol": args.symbol,
        "price_scale": spec.price_scale,
        "expected_price_range": [spec.expected_price_min, spec.expected_price_max],
        "range_start_utc": start.isoformat(),
        "range_end_utc": end.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_by": args.chunk_by,
        "rows_per_timeframe": rows_per_tf,
        "validation_overall_hard_failures": val.get("overall_hard_failures", -1),
        "files": {
            tf: str((sym_dir / tf / f"{args.symbol}_{tf}.parquet").relative_to(out_dir))
            for tf in TIMEFRAMES_MIN
        },
        "policy": ("Dukascopy is the single official data source. Do not "
                   "mix with other brokers. Do not interpolate missing "
                   "candles. Do not synthesise lower-timeframe data."),
    }
    (sym_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 5. cleanup raw
    if not args.keep_raw:
        raw = out_dir / "raw"
        if raw.exists():
            shutil.rmtree(raw, ignore_errors=True)
            print(f"[pipeline] cleaned raw/  (--keep-raw to disable)")

    # final summary block
    print()
    print("=" * 60)
    print("Dukascopy XAUUSD pipeline complete" if args.symbol == "XAUUSD"
          else f"Dukascopy {args.symbol} pipeline complete")
    print(f"  Symbol:     {args.symbol}")
    print(f"  Start:      {start.date()}")
    print(f"  End:        {end.date()}")
    print(f"  Timeframes generated: {sorted(rows_per_tf)}")
    print(f"  Rows per timeframe:")
    for tf in TIMEFRAMES_MIN:
        print(f"    {tf:3s} = {rows_per_tf.get(tf, 0):>10,}")
    print(f"  Validation overall_hard_failures: "
          f"{manifest['validation_overall_hard_failures']}")
    print(f"  Manifest:   {sym_dir / 'manifest.json'}")
    print(f"  Artifact path: {out_dir}/")
    print("=" * 60)
    return 0 if rc_v == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
