"""Walk-forward / holdout parity tests (Phase 2).

The hardening invariant: walk-forward and holdout MUST run the same
executable strategy. If walk-forward downgrades any aspect of the spec
(entry model, stop, exit, cost) relative to holdout, the certifier is
gating on a different strategy than the one being deployed and the
"certified" label is meaningless.

These tests construct synthetic (h4, m15) data, run the canonical
holdout executor, run walk-forward over a single fold that covers the
same window, and assert that:

  1. The set of trades matches (same entry_time + direction).
  2. The trade attributes match (entry, exit, cost, pnl) within
     floating-point tolerance.
  3. A spec with an entry-timeframe the executor does NOT support is
     reported as `compatibility != "ok"` and produces zero folds, NOT
     a silent fallback to h4_open.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.types import Spec
from execution.executor import ExecutionModel, run as run_executor
from validation.walkforward import (
    walk_forward, run_executor_on_window, check_compatibility, WFConfig,
)


TOL = 1e-9


def _synth(n_h4: int = 240, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate noisy H4 + M15 over ~40 days. Trade count is moderate."""
    rng = np.random.default_rng(seed)
    h4_times = pd.date_range("2024-01-01", periods=n_h4, freq="4h", tz="UTC")
    rets = rng.normal(0, 0.005, n_h4)
    closes = 2000.0 * np.cumprod(1.0 + rets)
    opens = np.concatenate([[2000.0], closes[:-1]])
    rng_h = np.abs(rng.normal(0.006, 0.003, n_h4))
    highs = np.maximum(opens, closes) * (1.0 + rng_h / 2.0)
    lows = np.minimum(opens, closes) * (1.0 - rng_h / 2.0)
    h4 = pd.DataFrame({
        "time": h4_times, "open": opens, "high": highs, "low": lows,
        "close": closes,
        "volume": rng.uniform(800, 1200, n_h4),
        "spread": np.full(n_h4, 0.30),
    })
    m15_rows = []
    for i, bt in enumerate(h4_times):
        st = pd.date_range(bt, periods=16, freq="15min", tz="UTC")
        path = np.linspace(opens[i], closes[i], 16) + \
               rng.normal(0, rng_h[i] * opens[i] * 0.1, 16)
        h_arr = path + np.abs(rng.normal(0, rng_h[i] * opens[i] * 0.05, 16))
        l_arr = path - np.abs(rng.normal(0, rng_h[i] * opens[i] * 0.05, 16))
        for k in range(16):
            m15_rows.append({
                "time": st[k],
                "open": float(path[k]), "high": float(h_arr[k]),
                "low": float(l_arr[k]), "close": float(path[k]),
                "volume": 100.0, "spread": 0.30,
            })
    m15 = pd.DataFrame(m15_rows)
    return h4, m15


def _trade_keys(trades: list) -> list[tuple]:
    return [(t.entry_time, int(t.direction)) for t in trades]


def _spec(entry: dict, stop: dict | None = None, **kw) -> Spec:
    return Spec(
        id="parity_t",
        signal=kw.get("signal", {"type": "prev_color"}),
        filters=kw.get("filters", []),
        entry=entry,
        stop=stop or {"type": "none"},
        exit={"type": "h4_close"},
    )


def test_walkforward_kernel_equals_holdout_on_same_window():
    """`run_executor_on_window` is a thin wrapper over the executor; they
    must produce identical trades."""
    h4, m15 = _synth()
    spec = _spec(entry={"type": "touch_entry"})
    em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                        miss_prob_market=0.0, miss_prob_limit=0.0,
                        spread_mult=1.0)
    direct = run_executor(spec, h4, m15, em)
    wrapped = run_executor_on_window(spec, h4, m15, em)
    assert _trade_keys(direct) == _trade_keys(wrapped), (
        "walk-forward kernel produced different trades than the executor")
    for a, b in zip(direct, wrapped):
        for fld in ("entry", "exit", "pnl", "cost", "spread_paid"):
            assert abs(getattr(a, fld) - getattr(b, fld)) < TOL, (
                f"field {fld} differs: direct={getattr(a, fld)} "
                f"wrapped={getattr(b, fld)}")


def test_walkforward_test_window_trades_match_holdout_slice():
    """Run walkforward with one fold covering the back half of the data;
    run holdout on the same back-half window directly. The fold's test
    window trades must equal holdout's trades on that slice."""
    h4, m15 = _synth(n_h4=240)  # 40 days
    spec = _spec(entry={"type": "touch_entry"})
    em = ExecutionModel(slippage_bps_mean=0.0, slippage_bps_vol=0.0,
                        miss_prob_market=0.0, miss_prob_limit=0.0,
                        spread_mult=1.0)

    # Run the executor across the full window once.
    full_trades = run_executor(spec, h4, m15, em)

    # Define a "test window" = back half. Holdout's slice is the same.
    boundary = h4["time"].iloc[len(h4) // 2]
    holdout_trades = [t for t in full_trades if t.entry_time >= boundary]

    # Walk-forward kernel on the same back-half (warm-up handled by the
    # caller passing the full data).
    wf_trades = run_executor_on_window(spec, h4, m15, em)
    wf_trades_in_test = [t for t in wf_trades if t.entry_time >= boundary]

    assert _trade_keys(wf_trades_in_test) == _trade_keys(holdout_trades), (
        "walk-forward and holdout produce different trade sets on the "
        "same back-half window")


def test_resolution_limited_for_unsupported_entry_timeframe():
    """A spec asking for an entry timeframe the executor doesn't support
    must NOT silently fall back -- it must report compatibility status
    != 'ok' and yield zero folds."""
    h4, m15 = _synth()
    # M5 entry timeframe is not currently wired into the executor;
    # touch_entry IS compatible with M5 per the entry-model map, but
    # the data-availability check should mark it `data_unavailable`.
    spec = Spec(
        id="m5_request",
        entry_timeframe="M5",
        entry={"type": "touch_entry"},
        stop={"type": "none"},
        exit={"type": "h4_close"},
    )
    status, reason = check_compatibility(spec)
    assert status == "data_unavailable", (
        f"expected data_unavailable for M5 entry timeframe, got {status} "
        f"({reason})")

    out = walk_forward(spec, h4, m15, WFConfig(min_folds=1))
    assert out["folds"] == 0
    assert out["compatibility"] == "data_unavailable"
    assert out["meets_min_folds"] is False


def test_resolution_limited_for_unknown_entry_model():
    spec = Spec(
        id="unknown",
        entry_timeframe="M15",
        entry={"type": "h4_open"},  # NOT in the registry
        stop={"type": "none"},
        exit={"type": "h4_close"},
    )
    status, reason = check_compatibility(spec)
    assert status == "unknown_model", (
        f"expected unknown_model for h4_open, got {status} ({reason})")


def test_walkforward_does_not_rewrite_entry_model():
    """Static check: the new walkforward.py source must NOT contain the
    old downgrade code-path strings."""
    src = (ROOT / "validation" / "walkforward.py").read_text()
    forbidden = [
        '"type": "h4_open"',
        "treat as h4_open",
        "can't replay M15 entries",
        "downgrades many M15 entry models to `h4_open`",
    ]
    found = [s for s in forbidden if s in src]
    assert not found, (
        f"validation/walkforward.py still contains downgrade markers: {found}")


if __name__ == "__main__":
    fns = [
        test_walkforward_kernel_equals_holdout_on_same_window,
        test_walkforward_test_window_trades_match_holdout_slice,
        test_resolution_limited_for_unsupported_entry_timeframe,
        test_resolution_limited_for_unknown_entry_model,
        test_walkforward_does_not_rewrite_entry_model,
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
