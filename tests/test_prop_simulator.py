"""Tests for the Phase-7 prop simulator hardening.

Lock in:
  * RiskModel.size() does not accept `trade_pnl_price` (compile-time
    check on the function signature).
  * size() output is invariant to the (synthetic) future PnL of the
    trade about to be sized — sizing depends only on pre-trade inputs.
  * Chronological replay groups trades by REAL UTC date and applies
    rules in order; outcome is deterministic for a given input.
  * Day-block bootstrap preserves intra-day clustering: a sampled
    block always contains a contiguous run of the same days.
  * Wilson CI bounds are sane: width shrinks with n; small-n CI is
    wide; CI sits inside [0, 1].
  * Verification status: real-firm with no `last_verified` ->
    `unverified`; with a recent date -> `verified`; with an old date
    -> `stale`; synthetic source -> `synthetic`.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prop_challenge.accounts import (
    AccountSpec, verification_status, can_certify_for_live,
    validate_schema, load_all,
)
from prop_challenge.challenge import (
    run_challenge, run_chronological_replay, _sample_day_blocks,
)
from prop_challenge.lockout import DailyRules
from prop_challenge.risk import RiskModel
from prop_challenge.stats import wilson_ci, bootstrap_median_ci


# ---------- signature / no-leakage ----------

def test_risk_size_signature_no_pnl():
    sig = inspect.signature(RiskModel.size)
    assert "trade_pnl_price" not in sig.parameters, (
        "RiskModel.size() must NOT accept trade_pnl_price; "
        f"got params {list(sig.parameters)}")
    # The hardening replacements ARE accepted.
    assert "stop_distance_price" in sig.parameters
    assert "atr_pre_entry" in sig.parameters


def test_risk_size_invariant_to_synthetic_future_pnl():
    """Calling size() twice with identical pre-trade inputs must give
    identical results. The signature has no future-PnL input, so this
    should be trivially deterministic."""
    rm = RiskModel(name="t", instrument="MGC", contracts_base=1,
                   dollar_risk_per_trade=50.0, contracts_max=5)
    base_kwargs = dict(
        balance=50_000, starting_balance=50_000, max_loss=2_500,
        last_trade_loss=False, equity_high=False,
        stop_distance_price=0.30, atr_pre_entry=0.50,
    )
    c1 = rm.size(**base_kwargs)
    c2 = rm.size(**base_kwargs)
    assert c1 == c2, f"deterministic sizing required; got c1={c1} c2={c2}"
    # And the formula: floor(50 / (0.30 * 10)) = 16, capped to contracts_max=5.
    expected_uncapped = int(50.0 / (0.30 * 10.0))
    assert c1 == min(5, expected_uncapped), (
        f"expected min(5, {expected_uncapped}) = {min(5, expected_uncapped)}, got {c1}")


def test_risk_size_responds_to_stop_distance():
    """Wider stop -> fewer contracts at the same dollar-risk target.
    Pick a dollar_risk small enough that the cap doesn't bind for
    either case so the comparison actually reflects the stop sizing."""
    rm = RiskModel(name="t", dollar_risk_per_trade=20.0, contracts_max=5)
    base = dict(balance=50_000, starting_balance=50_000, max_loss=2_500,
                last_trade_loss=False, equity_high=False, atr_pre_entry=None)
    # tight stop 0.30 -> per_contract_risk=$3 -> floor(20/3)=6, capped to 5
    # wide stop  2.00 -> per_contract_risk=$20 -> floor(20/20)=1
    tight = rm.size(stop_distance_price=0.30, **base)
    wide = rm.size(stop_distance_price=2.00, **base)
    assert tight > wide, (
        f"tighter stop should size MORE contracts ({tight}) "
        f"than wider stop ({wide})")


def test_risk_size_falls_back_to_base_without_stop():
    """If stop and ATR are unavailable, dollar-risk model can't size and
    falls back to contracts_base (which here is 1)."""
    rm = RiskModel(name="t", instrument="MGC", contracts_base=1,
                   dollar_risk_per_trade=50.0, contracts_max=5)
    c = rm.size(balance=50_000, starting_balance=50_000, max_loss=2_500,
                last_trade_loss=False, equity_high=False,
                stop_distance_price=None, atr_pre_entry=None)
    assert c == 1


# ---------- chronological replay ----------

def _spec_50k() -> AccountSpec:
    return AccountSpec(
        name="t_50k", firm="Test",
        starting_balance=50_000, profit_target=3_000,
        daily_loss_limit=1_500, max_loss=2_500, trailing_drawdown=2_500,
        drawdown_type="static", max_contracts=5,
        minimum_trading_days=2, consistency_rule_percent=50.0,
        payout_target=1_000, payout_min_days=2, max_challenge_days=10,
        source_url="synthetic", last_verified="2026-04-28",
    )


def _winning_trades(n_days: int = 4, per_day: int = 2) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=d)
        for i in range(per_day):
            rows.append({
                "entry_time": date + pd.Timedelta(hours=12 + i),
                "pnl": 60.0,                  # +$60 price-units per trade
                "stop_distance_price": 0.20,
            })
    return pd.DataFrame(rows)


def test_chronological_replay_passes_on_winning_ledger():
    spec = _spec_50k()
    rm = RiskModel(name="t", contracts_base=1, contracts_max=5)
    rules = DailyRules(name="none")
    df = _winning_trades(n_days=4, per_day=2)
    out = run_chronological_replay(df, spec, rm, rules, instrument="MGC")
    # 4 days × 2 trades × $60 × 1 contract × $10/pt - $0.5 fee/contract = ~$478
    # ... wait: pnl is in price units. $60 × 10 ($/pt) = $600 per trade × 8 = $4800.
    # But profit_target = $3000, so should pass after 5 trades.
    assert out["outcome"] == "pass", out
    assert out["days"] >= spec.minimum_trading_days


def test_chronological_replay_blows_up_on_big_loss_day():
    spec = _spec_50k()
    rm = RiskModel(name="t", contracts_base=1, contracts_max=5)
    rules = DailyRules(name="none")
    rows = []
    # Day 1: small win.
    rows.append({"entry_time": pd.Timestamp("2026-01-01 12:00", tz="UTC"),
                 "pnl": 50.0, "stop_distance_price": 0.20})
    # Day 2: huge loss > daily_loss_limit ($1500).
    rows.append({"entry_time": pd.Timestamp("2026-01-02 12:00", tz="UTC"),
                 "pnl": -200.0, "stop_distance_price": 0.20})  # -200 × 10 = -$2000
    df = pd.DataFrame(rows)
    out = run_chronological_replay(df, spec, rm, rules, instrument="MGC")
    assert out["outcome"] == "blowup", out
    assert out["breach"] in ("daily_loss", "max_loss_static")


def test_chronological_replay_groups_by_real_calendar_day():
    """Two trades on the same real day must count as one day, not two."""
    spec = _spec_50k()
    rm = RiskModel(name="t", contracts_base=1, contracts_max=5)
    rules = DailyRules(name="max1", max_trades_per_day=1)
    rows = [
        {"entry_time": pd.Timestamp("2026-01-01 09:00", tz="UTC"),
         "pnl": 50.0, "stop_distance_price": 0.20},
        # Second trade SAME day -> must be filtered by max1.
        {"entry_time": pd.Timestamp("2026-01-01 16:00", tz="UTC"),
         "pnl": 50.0, "stop_distance_price": 0.20},
        {"entry_time": pd.Timestamp("2026-01-02 09:00", tz="UTC"),
         "pnl": 50.0, "stop_distance_price": 0.20},
    ]
    df = pd.DataFrame(rows)
    out = run_chronological_replay(df, spec, rm, rules, instrument="MGC")
    # max1 + 2 unique days -> at most 2 trades total. End balance =
    # 50k + 2 × ($50 × 10 × 1 - 0.5) = 50k + 999 = 50,999.
    assert out["outcome"] == "timeout", out
    assert abs(out["end_balance"] - (50_000 + 2 * 499.5)) < 1.0


# ---------- day-block bootstrap ----------

def test_day_block_bootstrap_preserves_clustering():
    """A bootstrap draw with block_days=3 must contain at least one
    triplet of consecutive original day indices."""
    days = [(pd.Timestamp(f"2026-01-{d:02d}", tz="UTC"), pd.DataFrame())
            for d in range(1, 21)]
    rng = np.random.default_rng(42)
    drawn = _sample_day_blocks(days, block_days=3, n_days=12, rng=rng)
    # The drawn list is contiguous-block resampled from `days`; check
    # that ≥1 group of three adjacent (mod len) days appears in order.
    idx_map = {d[0]: i for i, d in enumerate(days)}
    drawn_idxs = [idx_map[d[0]] for d in drawn]
    found_triplet = False
    for i in range(len(drawn_idxs) - 2):
        a, b, c = drawn_idxs[i], drawn_idxs[i+1], drawn_idxs[i+2]
        if (b - a) % len(days) == 1 and (c - b) % len(days) == 1:
            found_triplet = True
            break
    assert found_triplet, "block_days=3 must produce at least one consecutive triplet"


# ---------- Wilson CI ----------

def test_wilson_ci_within_unit_interval():
    for s, n in [(0, 10), (5, 10), (10, 10), (1, 1000), (50, 100)]:
        p, lo, hi = wilson_ci(s, n)
        assert 0.0 <= lo <= p <= hi <= 1.0, (s, n, p, lo, hi)


def test_wilson_ci_shrinks_with_n():
    _, lo_small, hi_small = wilson_ci(50, 100)
    _, lo_big, hi_big = wilson_ci(500, 1000)
    width_small = hi_small - lo_small
    width_big = hi_big - lo_big
    assert width_big < width_small, (
        f"larger n must yield tighter CI; small n width {width_small}, "
        f"big n width {width_big}")


def test_wilson_ci_stays_in_zero_one_for_extremes():
    # 0% successes
    _, lo, hi = wilson_ci(0, 50)
    assert 0.0 <= lo and hi <= 1.0
    # 100% successes
    _, lo, hi = wilson_ci(50, 50)
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_median_ci_basic():
    arr = np.arange(1, 101, dtype=float)
    point, lo, hi = bootstrap_median_ci(arr, n_runs=500, seed=1)
    assert 40 < lo < 60
    assert 40 < point < 60
    assert 40 < hi < 65


# ---------- verification metadata ----------

def test_verification_status_branches():
    today = _dt.date(2026, 4, 28)

    real_unverified = AccountSpec(
        name="t", firm="x", starting_balance=50_000,
        profit_target=3000, daily_loss_limit=1500, max_loss=2500,
        trailing_drawdown=2500, drawdown_type="static",
        max_contracts=5, minimum_trading_days=5,
        consistency_rule_percent=50.0, payout_target=1000,
        payout_min_days=8, max_challenge_days=60,
        source_url="https://realfirm.example/", last_verified=None,
    )
    assert verification_status(real_unverified, today) == "unverified"

    real_recent = AccountSpec(**{**real_unverified.__dict__,
                                  "last_verified": "2026-03-01"})
    assert verification_status(real_recent, today) == "verified"

    real_stale = AccountSpec(**{**real_unverified.__dict__,
                                 "last_verified": "2025-01-01"})
    assert verification_status(real_stale, today) == "stale"

    synthetic = AccountSpec(**{**real_unverified.__dict__,
                                "source_url": "synthetic"})
    assert verification_status(synthetic, today) == "synthetic"


def test_can_certify_for_live_only_when_verified():
    today = _dt.date(2026, 4, 28)
    base = AccountSpec(
        name="t", firm="x", starting_balance=50_000,
        profit_target=3000, daily_loss_limit=1500, max_loss=2500,
        trailing_drawdown=2500, drawdown_type="static",
        max_contracts=5, minimum_trading_days=5,
        consistency_rule_percent=50.0, payout_target=1000,
        payout_min_days=8, max_challenge_days=60,
        source_url="https://realfirm.example/", last_verified=None,
    )
    assert can_certify_for_live(base, today) is False
    base_verified = AccountSpec(**{**base.__dict__, "last_verified": "2026-04-01"})
    assert can_certify_for_live(base_verified, today) is True


def test_schema_validate_rejects_missing_fields():
    bad = {"_meta": {"instrument_dollar_per_price_unit": {"MGC": 10.0}},
           "incomplete_acct": {"firm": "X"}}  # missing required fields
    raised = False
    try:
        validate_schema(bad)
    except ValueError as exc:
        raised = "missing required field" in str(exc)
    assert raised, "validate_schema must reject incomplete account configs"


def test_schema_validate_rejects_unknown_drawdown_type():
    bad = {"_meta": {"instrument_dollar_per_price_unit": {"MGC": 10.0}},
           "wrong_dd": {
               "firm": "X", "starting_balance": 50_000,
               "profit_target": 3000, "daily_loss_limit": 1500,
               "max_loss": 2500, "trailing_drawdown": 2500,
               "drawdown_type": "completely_made_up_type",
               "max_contracts": 5, "minimum_trading_days": 5,
               "consistency_rule_percent": 50.0, "payout_target": 1000,
               "payout_min_days": 8, "max_challenge_days": 60,
           }}
    raised = False
    try:
        validate_schema(bad)
    except ValueError as exc:
        raised = "drawdown_type" in str(exc)
    assert raised


def test_load_all_parses_real_config():
    accounts = load_all()
    # at least the topstep + mffu + generic families exist
    names = set(accounts.keys())
    assert any("topstep" in n for n in names)
    assert any("mffu" in n for n in names)
    assert any("generic" in n for n in names)
    # synthetic accounts are marked "synthetic"
    for name, spec in accounts.items():
        if "generic" in name:
            assert verification_status(spec) == "synthetic"


# ---------- end-to-end MC sanity ----------

def test_run_challenge_emits_ci_and_pass_rate_within_unit():
    """A small smoke run on a winning ledger must produce a passes
    probability with both CI bounds inside [0, 1]."""
    spec = _spec_50k()
    rm = RiskModel(name="t", contracts_base=1, contracts_max=5)
    rules = DailyRules(name="none")
    df = _winning_trades(n_days=20, per_day=2)
    cr = run_challenge(df, spec, rm, rules, n_runs=200,
                       instrument="MGC", seed=0)
    assert 0.0 <= cr.pass_probability_ci[0] <= cr.pass_probability \
                                                <= cr.pass_probability_ci[1] <= 1.0


if __name__ == "__main__":
    fns = [
        test_risk_size_signature_no_pnl,
        test_risk_size_invariant_to_synthetic_future_pnl,
        test_risk_size_responds_to_stop_distance,
        test_risk_size_falls_back_to_base_without_stop,
        test_chronological_replay_passes_on_winning_ledger,
        test_chronological_replay_blows_up_on_big_loss_day,
        test_chronological_replay_groups_by_real_calendar_day,
        test_day_block_bootstrap_preserves_clustering,
        test_wilson_ci_within_unit_interval,
        test_wilson_ci_shrinks_with_n,
        test_wilson_ci_stays_in_zero_one_for_extremes,
        test_bootstrap_median_ci_basic,
        test_verification_status_branches,
        test_can_certify_for_live_only_when_verified,
        test_schema_validate_rejects_missing_fields,
        test_schema_validate_rejects_unknown_drawdown_type,
        test_load_all_parses_real_config,
        test_run_challenge_emits_ci_and_pass_rate_within_unit,
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
