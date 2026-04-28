"""Run state + events scaffold tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.run_events import (
    EventWriter, emit_candidate, emit_stage, read_events, validate_event,
)
from core.run_state import (
    ALLOWED_STAGES, ALLOWED_STATUSES,
    Counts, RunState, make_run_id, next_seq,
)


def test_run_state_create_makes_directory_and_files():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        assert st.run_dir.exists()
        assert st.events_path.exists()
        assert st.progress_path.exists()
        # progress.json is valid JSON with the expected top-level keys
        payload = json.loads(st.progress_path.read_text())
        for k in ("run_id", "started_at", "status", "counts",
                  "active_candidates", "recent_events"):
            assert k in payload


def test_next_seq_increments_per_date():
    with TemporaryDirectory() as td:
        runs = Path(td)
        st1 = RunState.create(runs)
        date = st1.run_id.split("_")[0]
        st2 = RunState.create(runs)
        assert st1.run_id != st2.run_id
        assert next_seq(runs, date) == 3   # both already exist; next is 3


def test_set_stage_validates_against_allowed():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        st.set_stage("walk_forward")
        assert st.current_stage == "walk_forward"
        raised = False
        try:
            st.set_stage("totally_made_up_stage")
        except ValueError as exc:
            raised = "unknown stage" in str(exc)
        assert raised


def test_set_status_validates_against_allowed():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        st.set_status("running")
        st.set_status("certified")
        raised = False
        try:
            st.set_status("totally_made_up_status")
        except ValueError as exc:
            raised = "unknown status" in str(exc)
        assert raised


def test_bump_known_count_field():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        st.bump("generated", 5)
        st.bump("rejected_unavailable_data")
        assert st.counts.generated == 5
        assert st.counts.rejected_unavailable_data == 1
        raised = False
        try:
            st.bump("not_a_field")
        except ValueError:
            raised = True
        assert raised


def test_event_writer_appends_jsonl():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        ew = EventWriter(st)
        ew.emit({"stage": "ohlc_backtest", "status": "running",
                 "candidate_id": "c1"})
        ew.emit({"stage": "walk_forward", "status": "failed",
                 "candidate_id": "c1",
                 "failure_reasons": ["fail_walk_forward"]})
        events = read_events(st.events_path)
        assert len(events) == 2
        assert events[0]["candidate_id"] == "c1"
        assert events[0]["stage"] == "ohlc_backtest"
        # both events have an auto-stamped timestamp + run_id
        for e in events:
            assert e.get("timestamp")
            assert e.get("run_id") == st.run_id


def test_event_writer_rejects_bad_stage_or_status():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        ew = EventWriter(st)
        for bad in (
            {"stage": "not_a_stage", "status": "running"},
            {"stage": "walk_forward", "status": "weird_status"},
        ):
            raised = False
            try:
                ew.emit(bad)
            except ValueError:
                raised = True
            assert raised, f"expected ValueError for {bad}"


def test_progress_recent_events_capped():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        ew = EventWriter(st, keep_recent=3)
        for i in range(10):
            ew.emit({"stage": "ohlc_backtest", "status": "running",
                     "candidate_id": f"c{i}"})
        progress = json.loads(st.progress_path.read_text())
        assert len(progress["recent_events"]) <= 3


def test_emit_candidate_helper():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        ew = EventWriter(st)
        emit_candidate(ew, candidate_id="abc", family="session_sweep_reclaim",
                       stage="walk_forward", status="failed",
                       certification_level="research_only",
                       failure_reasons=["fail_walk_forward"],
                       metrics={"wf_pct_positive": 0.42},
                       message="Candidate failed walk-forward gate.")
        events = read_events(st.events_path)
        assert len(events) == 1
        e = events[0]
        assert e["candidate_id"] == "abc"
        assert e["family"] == "session_sweep_reclaim"
        assert e["certification_level"] == "research_only"
        assert e["failure_reasons"] == ["fail_walk_forward"]
        assert e["metrics"]["wf_pct_positive"] == 0.42


def test_emit_stage_helper():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        ew = EventWriter(st)
        emit_stage(ew, stage="leaderboard", status="passed",
                   message="leaderboard written",
                   metrics={"n_specs": 72})
        events = read_events(st.events_path)
        assert events[0]["stage"] == "leaderboard"
        assert events[0]["metrics"]["n_specs"] == 72


def test_emit_helpers_safe_with_none_writer():
    """Pipeline functions accept `events=None`; helpers must not crash."""
    out = emit_candidate(None, candidate_id="abc", family="x",
                         stage="walk_forward", status="failed")
    assert out is None
    out = emit_stage(None, stage="leaderboard", status="passed")
    assert out is None
    # but a malformed event still raises (catch misuse early)
    raised = False
    try:
        emit_stage(None, stage="not_a_stage", status="passed")
    except ValueError:
        raised = True
    assert raised


def test_write_summary_includes_extras():
    with TemporaryDirectory() as td:
        st = RunState.create(Path(td))
        st.set_status("certified")
        st.bump("certified", 3)
        st.write_summary({"git_head": "abc123",
                           "leaderboard": "results/x.csv"})
        payload = json.loads(st.summary_path.read_text())
        assert payload["status"] == "certified"
        assert payload["counts"]["certified"] == 3
        assert payload["extras"]["git_head"] == "abc123"


def test_validate_event_passes_well_formed():
    validate_event({"stage": "walk_forward", "status": "passed"})
    # minimal: stage / status both omitted is OK (writers can stamp later)
    validate_event({"message": "freeform"})


def test_run_id_format():
    rid = make_run_id()
    # YYYY-MM-DD_NNN
    assert len(rid) == len("2026-04-29_001")
    assert "_" in rid


if __name__ == "__main__":
    fns = [
        test_run_state_create_makes_directory_and_files,
        test_next_seq_increments_per_date,
        test_set_stage_validates_against_allowed,
        test_set_status_validates_against_allowed,
        test_bump_known_count_field,
        test_event_writer_appends_jsonl,
        test_event_writer_rejects_bad_stage_or_status,
        test_progress_recent_events_capped,
        test_emit_candidate_helper,
        test_emit_stage_helper,
        test_emit_helpers_safe_with_none_writer,
        test_write_summary_includes_extras,
        test_validate_event_passes_well_formed,
        test_run_id_format,
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
