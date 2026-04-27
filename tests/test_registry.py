"""Entry-model registry tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entry_models import registry


def test_registry_has_expected_models():
    expected = {
        "touch_entry", "reaction_close", "fib_limit_entry", "zone_midpoint_limit",
        "minor_structure_break", "delayed_entry_1", "delayed_entry_2", "sweep_reclaim",
    }
    have = set(registry.names())
    missing = expected - have
    assert not missing, f"missing entry models: {missing}"


def test_register_rejects_duplicates():
    try:
        registry.register("touch_entry")(lambda *a, **k: None)
    except ValueError:
        return
    raise AssertionError("registering a duplicate should have raised")


if __name__ == "__main__":
    test_registry_has_expected_models()
    print("  PASS  test_registry_has_expected_models")
    test_register_rejects_duplicates()
    print("  PASS  test_register_rejects_duplicates")
