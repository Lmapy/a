"""Pull the Dukascopy candle sidecar branch into data/dukascopy/candles/.

The build-dukascopy-data GitHub Actions workflow force-pushes the
candle parquet (per-year files + manifest + validation report) to a
sidecar branch named in its `commit_to_branch` input (default
data-dukascopy). This script downloads that branch as a codeload zip
(an allowlisted github.com path) and extracts the candle tree into
data/dukascopy/candles/. It does NOT touch any other branch.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "dukascopy" / "candles"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Lmapy/a")
    p.add_argument("--branch", default="data-dukascopy")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--out", default=str(TARGET))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    url = f"https://codeload.github.com/{args.repo}/zip/refs/heads/{args.branch}"
    print(f"[pull] fetching  {url}", flush=True)

    zip_path = ROOT / f"{Path(args.branch).name}.zip"
    with urlopen(url, timeout=120) as r, open(zip_path, "wb") as f:
        chunk = r.read(1 << 20)
        n = 0
        while chunk:
            f.write(chunk)
            n += len(chunk)
            chunk = r.read(1 << 20)
    print(f"[pull] downloaded {n / 1024 / 1024:.1f} MB to {zip_path}",
          flush=True)

    # Extract via stdlib unzip (avoids needing the OS `unzip` binary).
    import zipfile
    extract_to = ROOT / "_pull_tmp"
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_to)
    # codeload zips have a single top-level dir like a-data-dukascopy/
    top = next(extract_to.iterdir())
    src = top / "data" / "dukascopy" / "candles"
    if not src.exists():
        print(f"[pull] no candles in {src} — sidecar branch is empty?",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    print(f"[pull] wrote {out}", flush=True)

    # cleanup
    shutil.rmtree(extract_to, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    # Quick summary of what landed
    sym_dir = out / args.symbol
    if sym_dir.exists():
        for tf_dir in sorted(sym_dir.iterdir()):
            if tf_dir.is_dir():
                yrs = sorted(int(f.stem.split("=")[1])
                             for f in tf_dir.glob("year=*.parquet"))
                print(f"  {tf_dir.name:3s}  years={yrs}")
    print("[pull] done. now run:  make audit && make alpha && make prop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
