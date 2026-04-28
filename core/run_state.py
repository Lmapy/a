"""Run state — `progress.json` model.

A `RunState` represents one execution of a research pipeline. It
tracks aggregate counts and the current stage. Progress is written
to `results/runs/<run_id>/progress.json` so a future UI can read it
without intruding on the pipeline.

`progress.json` is overwritten atomically on each update. Per-event
detail goes to `events.jsonl` (see core/run_events.py).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---- canonical vocabulary --------------------------------------------------

ALLOWED_STAGES: tuple[str, ...] = (
    "strategy_generator",
    "feature_capability_auditor",
    "ohlc_backtest",
    "entry_model_lab",
    "walk_forward",
    "validation",
    "holdout",
    "risk_sweep",
    "daily_rule_optimiser",
    "prop_firm_simulator",
    "robustness_critic",
    "judge",
    "leaderboard",
    "report",
)

ALLOWED_STATUSES: tuple[str, ...] = (
    "queued",
    "running",
    "passed",
    "failed",
    "warning",
    "rejected",
    "watchlist",
    "candidate",
    "prop_candidate",
    "certified",
    "skipped",
)


# ---- run id helpers --------------------------------------------------------

_RUN_ID_RE = re.compile(r"^[\w\-:.]+$")


def make_run_id(now: _dt.datetime | None = None,
                seq: int | None = None) -> str:
    """Return a sortable run id of the form `2026-04-29_001`. The
    sequence number is auto-derived from existing runs in `runs_dir`
    if `seq` is None."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    base = now.strftime("%Y-%m-%d")
    if seq is None:
        seq = 1
    return f"{base}_{seq:03d}"


def next_seq(runs_dir: Path, date_str: str) -> int:
    """Find the next free sequence number for a date directory."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return 1
    seen = []
    for p in runs_dir.iterdir():
        if p.is_dir() and p.name.startswith(date_str + "_"):
            try:
                seen.append(int(p.name.split("_")[-1]))
            except ValueError:
                continue
    return (max(seen) + 1) if seen else 1


# ---- state object ----------------------------------------------------------

@dataclass
class Counts:
    generated: int = 0
    rejected_unavailable_data: int = 0
    rejected_broken: int = 0
    backtested: int = 0
    walk_forward_passed: int = 0
    walk_forward_failed: int = 0
    holdout_passed: int = 0
    holdout_failed: int = 0
    candidates: int = 0
    prop_candidates: int = 0
    certified: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class RunState:
    run_id: str
    runs_root: Path
    started_at: str = field(default_factory=lambda:
                             _dt.datetime.now(_dt.timezone.utc)
                             .isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda:
                             _dt.datetime.now(_dt.timezone.utc)
                             .isoformat(timespec="seconds"))
    status: str = "running"
    current_stage: str | None = None
    counts: Counts = field(default_factory=Counts)
    active_candidates: list[str] = field(default_factory=list)
    recent_events: list[dict] = field(default_factory=list)

    @classmethod
    def create(cls,
               runs_root: Path,
               run_id: str | None = None,
               *,
               status: str = "running") -> "RunState":
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
            run_id = make_run_id(seq=next_seq(runs_root, today))
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # ensure events.jsonl exists empty so a UI can tail it from t=0
        (run_dir / "events.jsonl").touch(exist_ok=True)
        st = cls(run_id=run_id, runs_root=runs_root, status=status)
        st.write_progress()
        return st

    # ---- mutations ----

    def set_stage(self, stage: str) -> None:
        if stage not in ALLOWED_STAGES:
            raise ValueError(
                f"unknown stage {stage!r}; allowed={ALLOWED_STAGES}")
        self.current_stage = stage
        self._touch()

    def set_status(self, status: str) -> None:
        if status not in ALLOWED_STATUSES:
            raise ValueError(
                f"unknown status {status!r}; allowed={ALLOWED_STATUSES}")
        self.status = status
        self._touch()

    def bump(self, field_name: str, by: int = 1) -> None:
        if not hasattr(self.counts, field_name):
            raise ValueError(
                f"unknown counts field {field_name!r}; allowed="
                f"{list(self.counts.__dataclass_fields__)}")
        setattr(self.counts, field_name, getattr(self.counts, field_name) + by)
        self._touch()

    def push_recent(self, event: dict, max_keep: int = 30) -> None:
        self.recent_events.append(event)
        if len(self.recent_events) > max_keep:
            self.recent_events = self.recent_events[-max_keep:]
        self._touch()

    def set_active(self, candidate_ids: list[str]) -> None:
        self.active_candidates = list(candidate_ids)
        self._touch()

    def _touch(self) -> None:
        self.updated_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    # ---- I/O ----

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.run_id

    @property
    def progress_path(self) -> Path:
        return self.run_dir / "progress.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    def to_json(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "current_stage": self.current_stage,
            "counts": asdict(self.counts),
            "active_candidates": list(self.active_candidates),
            "recent_events": list(self.recent_events),
        }

    def write_progress(self) -> None:
        """Atomic write so a UI reading the file never sees a half-written
        JSON document."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_json(), indent=2, default=str)
        # write to temp in same dir then replace
        fd, tmp = tempfile.mkstemp(prefix=".progress.", dir=str(self.run_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, self.progress_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def write_summary(self, extras: dict | None = None) -> None:
        """Once-per-run final summary. Combines progress with caller-
        provided extras (typically commit SHA, paths to leaderboards,
        runtime, etc.)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        if extras:
            payload["extras"] = extras
        self.summary_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
