"""Tests for the executor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Spec
from data.loader import load_all
from execution.executor import ExecutionModel, run as run_exec


def test_executor_runs_and_records_excursions():
    ds = load_all()
    spec = Spec(
        id="t",
        filters=[{"type": "regime", "ma_n": 50, "side": "with"}],
        entry={"type": "fib_limit_entry", "level": 0.5},
        stop={"type": "prev_h4_open"},
        exit={"type": "h4_close"},
    )
    trades = run_exec(spec, ds["h4"], ds["m15"], ExecutionModel())
    assert len(trades) > 0
    for t in trades:
        assert t.mae >= 0 and t.mfe >= 0
        assert t.entry_time <= t.exit_time
        assert t.fill_kind in ("market", "limit")
        assert t.h4_bucket is not None


def test_stress_reduces_returns():
    ds = load_all()
    spec = Spec(
        id="t",
        filters=[{"type": "regime", "ma_n": 50, "side": "with"}],
        entry={"type": "touch_entry"},
        stop={"type": "prev_h4_open"},
        exit={"type": "h4_close"},
    )
    base = run_exec(spec, ds["h4"], ds["m15"], ExecutionModel())
    stressed = run_exec(spec, ds["h4"], ds["m15"], ExecutionModel().stress())
    base_pnl = sum(t.pnl for t in base)
    stress_pnl = sum(t.pnl for t in stressed)
    assert stress_pnl <= base_pnl, f"stress increased pnl: base={base_pnl}, stress={stress_pnl}"


if __name__ == "__main__":
    test_executor_runs_and_records_excursions()
    print("  PASS  test_executor_runs_and_records_excursions")
    test_stress_reduces_returns()
    print("  PASS  test_stress_reduces_returns")
