"""PropCandidate schema + Spec adaptor tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.candidate import (
    AccountRef, DailyRulesBlock, PropCandidate, RiskBlock,
    reject_unavailable_data,
)
from core.certification import CertificationLevel, FailureReason
from core.types import Spec


def _make() -> PropCandidate:
    return PropCandidate(
        id="t_001",
        family="session_sweep_reclaim",
        signal={"type": "prev_color"},
        filters=[{"type": "body_atr", "min": 0.5, "atr_n": 14}],
        entry={"type": "touch_entry"},
        stop={"type": "prev_h4_open"},
        exit={"type": "h4_close"},
        risk=RiskBlock(name="dollar_risk_50", dollar_risk_per_trade=50.0),
        daily_rules=DailyRulesBlock(name="max2", max_trades_per_day=2),
        account=AccountRef(name="topstep_50k"),
    )


def test_to_spec_returns_executor_compatible_spec():
    cand = _make()
    spec = cand.to_spec()
    assert isinstance(spec, Spec)
    assert spec.id == "t_001"
    assert spec.signal["type"] == "prev_color"
    assert spec.entry["type"] == "touch_entry"
    assert spec.stop["type"] == "prev_h4_open"
    assert spec.exit["type"] == "h4_close"
    assert spec.cost_bps == 1.5    # default propagated
    # the rich risk/daily/account live on the candidate, not on the Spec
    assert spec.risk.get("name") == "dollar_risk_50"


def test_from_spec_round_trips_strategy_logic():
    cand = _make()
    spec = cand.to_spec()
    reborn = PropCandidate.from_spec(
        spec, family="session_sweep_reclaim",
        risk=cand.risk, daily_rules=cand.daily_rules, account=cand.account,
    )
    assert reborn.id == cand.id
    assert reborn.signal == cand.signal
    assert reborn.filters == cand.filters
    assert reborn.entry == cand.entry
    assert reborn.stop == cand.stop
    assert reborn.exit == cand.exit
    assert reborn.risk.name == cand.risk.name
    assert reborn.daily_rules.name == cand.daily_rules.name
    assert reborn.account.name == cand.account.name


def test_json_round_trip_preserves_certification_state():
    cand = _make()
    cand = reject_unavailable_data(cand,
                                    unavailable_features=["vwap"],
                                    unavailable_tokens=["vwap_dist"])
    payload = cand.to_json()
    s = json.dumps(payload)
    reborn = PropCandidate.from_json(json.loads(s))
    assert reborn.id == cand.id
    assert reborn.certification_level == \
        CertificationLevel.REJECTED_UNAVAILABLE_DATA
    assert FailureReason.REJECTED_UNAVAILABLE_DATA in reborn.failure_reasons
    assert reborn.rejection_detail.get("unavailable_features") == ["vwap"]


def test_default_cert_level_is_candidate():
    cand = _make()
    assert cand.certification_level == CertificationLevel.CANDIDATE
    assert cand.failure_reasons == []


def test_to_json_str_returns_compact_string():
    s = _make().to_json_str()
    assert isinstance(s, str)
    # compact form: no spaces after commas / colons
    assert ", " not in s and ": " not in s
    obj = json.loads(s)
    assert obj["id"] == "t_001"


def test_blocks_round_trip_through_dict():
    risk = RiskBlock(dollar_risk_per_trade=75.0, contracts_max=3)
    assert RiskBlock.from_json(risk.to_json()) == risk
    daily = DailyRulesBlock(max_trades_per_day=2, stop_after_n_losses=1)
    assert DailyRulesBlock.from_json(daily.to_json()) == daily
    acct = AccountRef(name="mffu_pro_50k", instrument="MGC", notes="x")
    assert AccountRef.from_json(acct.to_json()) == acct


if __name__ == "__main__":
    fns = [
        test_to_spec_returns_executor_compatible_spec,
        test_from_spec_round_trips_strategy_logic,
        test_json_round_trip_preserves_certification_state,
        test_default_cert_level_is_candidate,
        test_to_json_str_returns_compact_string,
        test_blocks_round_trip_through_dict,
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
