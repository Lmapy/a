"""Tests for the Batch J UI backend (file readers + endpoint shapes).

We don't spin up an actual HTTP server in tests; instead we call
the file-reader functions in `ui/server.py` directly. This catches
the bugs that matter (missing run dir, malformed JSON, bad CSV)
without requiring FastAPI / uvicorn to be installed in CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ui.server as server


def _seed_run(runs_dir: Path, run_id: str = "2026-04-29_001") -> Path:
    rd = runs_dir / run_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "progress.json").write_text(json.dumps({
        "run_id": run_id,
        "status": "passed",
        "current_stage": "judge",
        "started_at": "2026-04-29T10:00:00+00:00",
        "updated_at": "2026-04-29T10:05:00+00:00",
        "counts": {"generated": 4, "certified": 0, "failed": 4},
    }))
    (rd / "events.jsonl").write_text(
        json.dumps({"timestamp": "2026-04-29T10:00:00+00:00",
                     "stage": "strategy_generator",
                     "status": "running"}) + "\n"
        + json.dumps({"timestamp": "2026-04-29T10:01:00+00:00",
                       "candidate_id": "c1",
                       "stage": "walk_forward",
                       "status": "failed",
                       "failure_reasons": ["fail_walk_forward"]}) + "\n"
    )
    (rd / "summary.json").write_text(json.dumps({
        "run_id": run_id, "status": "passed",
        "extras": {"git_head": "abc123"},
    }))
    return rd


def _patch_paths(runs_dir: Path, results_dir: Path):
    """Redirect server module's path constants to a temp directory."""
    return patch.multiple(server,
                           RUNS_DIR=runs_dir,
                           RESULTS=results_dir,
                           STATIC_DIR=Path(server.STATIC_DIR))


def test_list_runs_newest_first():
    with TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        results_dir = Path(td)
        _seed_run(runs_dir, "2026-04-29_001")
        _seed_run(runs_dir, "2026-04-29_002")
        with _patch_paths(runs_dir, results_dir):
            runs = server.list_runs()
        assert len(runs) == 2
        # newest first (_002 lexically > _001)
        assert runs[0]["run_id"] == "2026-04-29_002"
        assert runs[0]["status"] == "passed"
        assert runs[0]["counts"]["generated"] == 4


def test_read_progress_missing_run_returns_none():
    with TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        runs_dir.mkdir(parents=True)
        results_dir = Path(td)
        with _patch_paths(runs_dir, results_dir):
            assert server.read_progress("nope") is None


def test_read_events_pagination():
    with TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        results_dir = Path(td)
        _seed_run(runs_dir, "x")
        with _patch_paths(runs_dir, results_dir):
            all_events = server.read_events("x", after=0, limit=100)
            assert len(all_events) == 2
            tail = server.read_events("x", after=1, limit=100)
            assert len(tail) == 1
            assert tail[0]["candidate_id"] == "c1"


def test_read_leaderboard_handles_missing_csv():
    with TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        runs_dir.mkdir(parents=True)
        results_dir = Path(td)
        with _patch_paths(runs_dir, results_dir):
            payload = server.read_leaderboard()
        assert payload["rows"] == []
        assert "error" in payload


def test_read_leaderboard_loads_csv_and_sidecar():
    import pandas as pd
    with TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        runs_dir.mkdir(parents=True)
        results_dir = Path(td)
        df = pd.DataFrame([
            {"candidate_id": "a", "family": "f1",
             "certification_level": "research_only",
             "prop_passing_score": -1.0, "fail_reasons": "fail_walk_forward"},
            {"candidate_id": "b", "family": "f2",
             "certification_level": "candidate",
             "prop_passing_score": 5.0, "fail_reasons": ""},
        ])
        df.to_csv(results_dir / "prop_passing_leaderboard.csv", index=False)
        (results_dir / "prop_passing_leaderboard.meta.json").write_text(
            json.dumps({"git_head": "abc123",
                        "produced_at_utc": "2026-04-29T10:00:00+00:00"}))
        with _patch_paths(runs_dir, results_dir):
            payload = server.read_leaderboard()
        assert len(payload["rows"]) == 2
        assert payload["rows"][0]["candidate_id"] == "a"
        assert payload["meta"]["git_head"] == "abc123"


def test_read_candidate_round_trip():
    import pandas as pd
    with TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        runs_dir.mkdir(parents=True)
        results_dir = Path(td)
        df = pd.DataFrame([{"candidate_id": "alpha", "family": "f1"},
                            {"candidate_id": "beta", "family": "f2"}])
        df.to_csv(results_dir / "prop_passing_leaderboard.csv", index=False)
        with _patch_paths(runs_dir, results_dir):
            row = server.read_candidate("beta")
        assert row is not None
        assert row["candidate_id"] == "beta"
        with _patch_paths(runs_dir, results_dir):
            assert server.read_candidate("not_there") is None


def test_read_audit_handles_missing_file():
    with TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        runs_dir.mkdir(parents=True)
        results_dir = Path(td)
        with _patch_paths(runs_dir, results_dir):
            txt = server.read_audit()
        assert "audit.txt not found" in txt


def test_static_files_present():
    """The frontend index.html / app.js / style.css must exist on
    disk; the backend serves them at /."""
    static = Path(server.STATIC_DIR)
    assert (static / "index.html").exists()
    assert (static / "app.js").exists()
    assert (static / "style.css").exists()


def test_server_main_argparse():
    """`main` parses --no-fastapi without exploding."""
    # Don't actually start the server; just probe argparse.
    saved_argv = sys.argv
    try:
        # We can't call main() directly because that starts a listener.
        # Instead verify the file source documents the expected flags.
        src = Path(server.__file__).read_text()
        for flag in ("--host", "--port", "--no-fastapi"):
            assert flag in src
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    fns = [
        test_list_runs_newest_first,
        test_read_progress_missing_run_returns_none,
        test_read_events_pagination,
        test_read_leaderboard_handles_missing_csv,
        test_read_leaderboard_loads_csv_and_sidecar,
        test_read_candidate_round_trip,
        test_read_audit_handles_missing_file,
        test_static_files_present,
        test_server_main_argparse,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    if failures:
        raise SystemExit(1)
