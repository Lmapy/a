"""Tests for the entry-model timeframe compatibility map."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entry_models.compatibility import (
    ENTRY_MODEL_TIMEFRAME_MAP,
    compatibility_status,
    is_compatible,
)
from entry_models import registry


def test_map_covers_every_registered_entry_model():
    for name in registry.names():
        assert name in ENTRY_MODEL_TIMEFRAME_MAP, \
            f"missing compatibility entry for: {name}"


def test_sweep_reclaim_only_low_timeframes():
    assert is_compatible("sweep_reclaim", "M1")
    assert is_compatible("sweep_reclaim", "M5")
    assert not is_compatible("sweep_reclaim", "M15")
    assert not is_compatible("sweep_reclaim", "H1")


def test_fib_limit_entry_works_on_m5_to_m30():
    assert is_compatible("fib_limit_entry", "M5")
    assert is_compatible("fib_limit_entry", "M15")
    assert is_compatible("fib_limit_entry", "M30")
    assert not is_compatible("fib_limit_entry", "M1")


def test_status_codes():
    assert compatibility_status("touch_entry", "M15", True) == "ok"
    assert compatibility_status("touch_entry", "M1", True) == "resolution_limited"
    assert compatibility_status("touch_entry", "M15", False) == "data_unavailable"
    assert compatibility_status("nonexistent_model", "M15", True) == "unknown_model"


if __name__ == "__main__":
    test_map_covers_every_registered_entry_model()
    print("  PASS  test_map_covers_every_registered_entry_model")
    test_sweep_reclaim_only_low_timeframes()
    print("  PASS  test_sweep_reclaim_only_low_timeframes")
    test_fib_limit_entry_works_on_m5_to_m30()
    print("  PASS  test_fib_limit_entry_works_on_m5_to_m30")
    test_status_codes()
    print("  PASS  test_status_codes")
