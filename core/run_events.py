"""Run events — append-only `events.jsonl` writer.

Each event is one JSON object on its own line. Format:

    {
      "timestamp": "2026-04-29T10:42:01Z",
      "run_id": "2026-04-29_001",
      "candidate_id": "london_sweep_reclaim_017",
      "family": "session_sweep_reclaim",
      "stage": "walk_forward",
      "status": "failed",
      "certification_level": "research_only",
      "failure_reasons": ["fail_walk_forward"],
      "metrics": {"wf_pct_positive": 0.42, "wf_median_sharpe": -0.18},
      "message": "Candidate failed walk-forward gate."
    }

Design rules
------------
* Append-only. Never rewrite or truncate `events.jsonl`.
* Each event line is a self-contained valid JSON object so a UI can
  tail the file and parse line-by-line without state.
* No file lock; Python's append-mode write is atomic for typical
  event sizes (<4 KB) on the platforms we care about. If multiple
  processes need to write to the same file, switch to a queue.
* Optional. The hardened pipeline must still work if no EventWriter
  is constructed. Pipeline functions accept an `events: EventWriter |
  None` argument; helpers (`emit`, `emit_candidate`) are no-ops when
  the writer is None.

Usage
-----
    from core.run_state import RunState
    from core.run_events import EventWriter, emit_candidate

    state = RunState.create(runs_root=Path("results/runs"))
    events = EventWriter(state)

    state.set_stage("walk_forward")
    emit_candidate(events, candidate_id="abc",
                   family="session_sweep_reclaim",
                   stage="walk_forward",
                   status="failed",
                   certification_level="research_only",
                   failure_reasons=["fail_walk_forward"],
                   metrics={"wf_pct_positive": 0.42},
                   message="Candidate failed walk-forward gate.")
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.run_state import (
    ALLOWED_STAGES, ALLOWED_STATUSES, RunState,
)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def validate_event(event: dict) -> None:
    """Raise ValueError if the event is malformed.

    Required keys: stage, status. Optional keys are passed through
    untouched so the schema can grow without breaking existing
    consumers.
    """
    if not isinstance(event, dict):
        raise ValueError("event must be a dict")
    stage = event.get("stage")
    status = event.get("status")
    if stage is not None and stage not in ALLOWED_STAGES:
        raise ValueError(f"unknown stage {stage!r}; allowed={ALLOWED_STAGES}")
    if status is not None and status not in ALLOWED_STATUSES:
        raise ValueError(f"unknown status {status!r}; allowed={ALLOWED_STATUSES}")


@dataclass
class EventWriter:
    """Append events to `events.jsonl` and refresh `progress.json`.

    Use `EventWriter(state)` after creating a RunState. The writer
    holds a reference to the state so it can update aggregate counts
    and recent_events after every emission.
    """
    state: RunState
    update_progress_each_event: bool = True
    keep_recent: int = 30

    def emit(self, event: dict) -> dict:
        """Validate, timestamp, append, optionally update progress.
        Returns the (possibly enriched) event."""
        validate_event(event)
        enriched = dict(event)
        enriched.setdefault("timestamp", _now_iso())
        enriched.setdefault("run_id", self.state.run_id)
        line = json.dumps(enriched, default=str, separators=(",", ":"))
        with self.state.events_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.state.push_recent({k: enriched.get(k)
                                 for k in ("timestamp", "candidate_id",
                                            "stage", "status",
                                            "certification_level")},
                                max_keep=self.keep_recent)
        if self.update_progress_each_event:
            self.state.write_progress()
        return enriched


# ---- thin convenience helpers ----------------------------------------------

def _opt_emit(writer: EventWriter | None, event: dict) -> dict | None:
    """Emit through the writer if present; do nothing if None."""
    if writer is None:
        validate_event(event)   # still validate so misuse fails loudly
        return None
    return writer.emit(event)


def emit_candidate(writer: EventWriter | None,
                   *,
                   candidate_id: str,
                   family: str,
                   stage: str,
                   status: str,
                   certification_level: str | None = None,
                   failure_reasons: list[str] | None = None,
                   metrics: dict[str, Any] | None = None,
                   message: str | None = None) -> dict | None:
    """Standardised candidate-level event."""
    event = {
        "candidate_id": candidate_id,
        "family": family,
        "stage": stage,
        "status": status,
    }
    if certification_level is not None:
        event["certification_level"] = certification_level
    if failure_reasons:
        event["failure_reasons"] = list(failure_reasons)
    if metrics:
        event["metrics"] = dict(metrics)
    if message:
        event["message"] = message
    return _opt_emit(writer, event)


def emit_stage(writer: EventWriter | None,
               *,
               stage: str,
               status: str,
               message: str | None = None,
               metrics: dict[str, Any] | None = None) -> dict | None:
    """Stage-level event (no candidate). Useful for "stage X started",
    "leaderboard written"."""
    event = {"stage": stage, "status": status}
    if message:
        event["message"] = message
    if metrics:
        event["metrics"] = dict(metrics)
    return _opt_emit(writer, event)


def read_events(events_path: Path) -> list[dict]:
    """Load every event from a run's events.jsonl. Useful for tests
    and for the future UI bootstrap."""
    events_path = Path(events_path)
    if not events_path.exists():
        return []
    out = []
    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
