"""Cross-platform test runner.

`make test` is the canonical entrypoint on Linux/Mac/WSL. On Windows
without GNU make installed, use:

    python3 scripts/run_tests.py

Each test file is run as a script (its __main__ block prints PASS lines
and raises SystemExit(1) on the first failure). The runner exits 0 only
if every file exits 0.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEST_FILES = [
    "tests/test_executor.py",
    "tests/test_registry.py",
    "tests/test_compatibility.py",
    "tests/test_stop_exit_spread.py",
    "tests/test_validator.py",
    "tests/test_splits.py",
    "tests/test_no_lookahead.py",
    "tests/test_walkforward_parity.py",
    "tests/test_statistical_tests.py",
    "tests/test_trade_metrics.py",
    "tests/test_prop_simulator.py",
    "tests/test_feature_capability.py",
    "tests/test_certification.py",
    "tests/test_candidate.py",
    "tests/test_tpo.py",
    "tests/test_run_events.py",
    "tests/test_strategies.py",
    "tests/test_batch_h.py",
    "tests/test_ui.py",
]


def main() -> int:
    failed: list[str] = []
    for rel in TEST_FILES:
        path = ROOT / rel
        if not path.exists():
            print(f"  SKIP   {rel}  (file missing)")
            continue
        print(f"=== {rel} ===")
        rc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT)).returncode
        if rc != 0:
            failed.append(rel)
            print(f"  ABORT  {rel}  (exit {rc})")
    print()
    if failed:
        print(f"FAIL  {len(failed)} test file(s) failed:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"PASS  all {len(TEST_FILES)} test files passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
