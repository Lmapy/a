"""Feature capability tests.

Verify:
  * VWAP / Volume Profile / footprint / delta / CVD / orderbook
    tokens are rejected against the canonical (OHLC-only) registry.
  * TPO tokens are allowed (TPO does not require real volume).
  * Existing OHLC tokens used by the hardened executor are allowed.
  * Unknown tokens are reported but treated as OHLC-only by default.
  * The classifier finds offending tokens in signal/filters/entry/stop/exit.
  * Suggested replacements are surfaced for VWAP / volume_profile inputs.
  * `assert_only_ohlc_only` raises on a bad candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.feature_capability import (
    CANONICAL_REGISTRY, CapabilityRegistry,
    assert_only_ohlc_only, classify_candidate, is_available,
    STRATEGY_FEATURE_TAGS,
)


def _spec(**parts):
    """Build a minimal spec-shaped dict."""
    base = {"signal": {"type": "prev_color"}, "filters": [],
            "entry": {"type": "touch_entry"},
            "stop": {"type": "prev_h4_open"},
            "exit": {"type": "h4_close"}}
    base.update(parts)
    return base


def test_canonical_registry_disables_real_volume_features():
    r = CANONICAL_REGISTRY
    assert r.ohlc is True and r.tpo is True
    for off in ("real_volume", "bid_ask_volume", "footprint", "delta",
                "cvd", "orderbook", "vwap", "volume_profile"):
        assert getattr(r, off) is False, f"{off} must be disabled by default"


def test_vwap_filter_rejected():
    spec = _spec(filters=[{"type": "vwap_dist", "max_z": 2.0}])
    v = classify_candidate(spec)
    assert v.status == "rejected_unavailable_data"
    assert "vwap" in v.unavailable_features
    assert "vwap_dist" in v.unavailable_tokens
    # rename hint should map vwap_dist -> atr_distance_from_session_mean
    assert v.rename_hints.get("vwap_dist") == "atr_distance_from_session_mean"


def test_volume_profile_rejected():
    spec = _spec(filters=[{"type": "volume_poc"}])
    v = classify_candidate(spec)
    assert v.status == "rejected_unavailable_data"
    assert "volume_profile" in v.unavailable_features
    assert v.rename_hints.get("volume_poc") == "tpo_poc"


def test_footprint_delta_cvd_orderbook_rejected():
    for token, feat in [
        ("footprint", "footprint"),
        ("delta", "delta"),
        ("cvd", "cvd"),
        ("bid_ask_imbalance", "bid_ask_volume"),
        ("dom_liquidity", "orderbook"),
    ]:
        spec = _spec(filters=[{"type": token}])
        v = classify_candidate(spec)
        assert v.status == "rejected_unavailable_data", \
            f"{token} should be rejected, got {v.status}"
        assert feat in v.unavailable_features


def test_tpo_tokens_allowed_without_volume():
    """TPO is time-at-price; available on OHLC alone. The canonical
    registry must let tpo_poc / tpo_vah / tpo_val through."""
    spec = _spec(
        filters=[{"type": "tpo_value_acceptance"},
                  {"type": "tpo_poor_high"}],
        entry={"type": "tpo_value_rejection"},
    )
    v = classify_candidate(spec)
    assert v.status == "ok", (v.status, v.unavailable_tokens)


def test_existing_ohlc_filters_still_allowed():
    spec = _spec(filters=[
        {"type": "body_atr"},
        {"type": "regime", "ma_n": 50, "side": "with"},
        {"type": "session", "hours_utc": [12, 13]},
        {"type": "atr_percentile"},
        {"type": "pdh_pdl"},
        {"type": "regime_class"},
        {"type": "wick_ratio"},
    ])
    v = classify_candidate(spec)
    assert v.status == "ok", (v.status, v.unavailable_tokens)


def test_unknown_token_not_rejected_but_reported():
    spec = _spec(filters=[{"type": "totally_made_up_filter_xyz"}])
    v = classify_candidate(spec)
    assert v.status == "ok"
    assert "totally_made_up_filter_xyz" in v.unknown_tokens


def test_assert_only_ohlc_only_raises_on_unavailable():
    spec = _spec(filters=[{"type": "vwap_dist"}])
    raised = False
    try:
        assert_only_ohlc_only(spec)
    except ValueError as exc:
        raised = "unavailable" in str(exc).lower()
    assert raised


def test_assert_only_ohlc_only_passes_for_clean_spec():
    spec = _spec()
    assert_only_ohlc_only(spec)   # no exception


def test_is_available_helper():
    assert is_available("touch_entry") is True
    assert is_available("tpo_poc") is True
    assert is_available("vwap") is False
    assert is_available("delta") is False
    assert is_available("totally_made_up") is True   # unknown -> OHLC


def test_capability_can_be_overridden_for_a_dataset_with_real_volume():
    """If a future dataset adds real volume, the registry can declare
    it. Same candidate must then be accepted."""
    custom = CapabilityRegistry(real_volume=True, vwap=True,
                                 volume_profile=True)
    spec = _spec(filters=[{"type": "vwap_dist"}, {"type": "volume_poc"}])
    v = classify_candidate(spec, registry=custom)
    assert v.status == "ok", (v.status, v.unavailable_tokens)


def test_token_tags_have_no_typos_in_required_features():
    """Sanity check the tag map: every required feature must be a
    known field on `CapabilityRegistry`."""
    valid = set(CapabilityRegistry.__dataclass_fields__)
    for token, reqs in STRATEGY_FEATURE_TAGS.items():
        for r in reqs:
            assert r in valid, (
                f"token {token!r} declares requirement {r!r} that is "
                f"not on CapabilityRegistry; valid={sorted(valid)}")


if __name__ == "__main__":
    fns = [
        test_canonical_registry_disables_real_volume_features,
        test_vwap_filter_rejected,
        test_volume_profile_rejected,
        test_footprint_delta_cvd_orderbook_rejected,
        test_tpo_tokens_allowed_without_volume,
        test_existing_ohlc_filters_still_allowed,
        test_unknown_token_not_rejected_but_reported,
        test_assert_only_ohlc_only_raises_on_unavailable,
        test_assert_only_ohlc_only_passes_for_clean_spec,
        test_is_available_helper,
        test_capability_can_be_overridden_for_a_dataset_with_real_volume,
        test_token_tags_have_no_typos_in_required_features,
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
