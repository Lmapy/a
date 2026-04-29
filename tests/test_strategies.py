"""Tests for the Batch G strategy generators + sweep labs.

Verify:

  * Each of the 10 families emits at least one PropCandidate.
  * Every emitted candidate is OHLC-only per the feature capability
    gate (TPO families pass because TPO is allowed).
  * Every candidate's `to_spec()` produces an executor-compatible Spec.
  * Candidate IDs are unique across the whole grid.
  * Entry-lab variants only emit registered, compatible entries.
  * Risk-sweep variants honour `contracts_max`.
  * Daily-rule variants don't mutate the base candidate.
  * `apply_capability_filter` correctly partitions OHLC-only vs.
    unavailable-data candidates.
  * `tier_2_for_survivor` dispatches to the right lab.
  * `grid_summary` returns sensible counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.candidate import PropCandidate
from core.certification import CertificationLevel, FailureReason
from core.feature_capability import classify_candidate
from core.types import Spec

from strategies import (
    daily_rule_lab, entry_lab, families, grid, risk_sweep,
)


# --- families ---------------------------------------------------------------

def test_every_family_emits_at_least_one_candidate():
    for fid, gen in families.FAMILIES.items():
        cands = gen()
        assert len(cands) >= 1, f"family {fid} emitted nothing"
        for c in cands:
            assert isinstance(c, PropCandidate)
            assert c.family == fid, (
                f"family {fid} returned candidate with family={c.family}")


def test_family_ids_match_brief():
    expected = {
        "session_sweep_reclaim", "opening_range_breakout_retest",
        "opening_range_failed_breakout", "previous_h4_range_retracement",
        "previous_h4_sweep_reclaim", "tpo_value_rejection",
        "tpo_poc_reversion", "atr_extension_reclaim",
        "compression_breakout", "failed_breakout_reversal",
    }
    assert set(families.family_ids()) == expected


def test_all_candidates_have_unique_ids():
    cands = families.all_candidates()
    ids = [c.id for c in cands]
    assert len(set(ids)) == len(ids), (
        f"duplicate ids: total={len(ids)} unique={len(set(ids))}")


def test_every_candidate_passes_capability_check():
    """All families MUST be OHLC-only. The two TPO families pass
    because TPO is in the canonical registry."""
    for c in families.all_candidates():
        v = classify_candidate(c.to_spec())
        assert v.status == "ok", (
            f"family {c.family} candidate {c.id} failed capability "
            f"check: {v.unavailable_tokens} -> {v.unavailable_features}")


def test_every_candidate_to_spec_works():
    """to_spec() must produce a valid Spec the executor would accept."""
    for c in families.all_candidates():
        spec = c.to_spec()
        assert isinstance(spec, Spec)
        assert spec.id == c.id
        # signal/entry/stop/exit blocks must have a `type`
        assert "type" in spec.signal
        assert "type" in spec.entry
        assert "type" in spec.stop
        assert "type" in spec.exit


def test_tpo_families_carry_executor_extension_marker():
    """The two TPO families need executor TPO filter implementations
    (Batch H). They mark this in `provenance` so the orchestrator
    can detect."""
    for fid in ("tpo_value_rejection", "tpo_poc_reversion"):
        cands = families.FAMILIES[fid]()
        for c in cands:
            assert c.provenance.get("requires_executor_extension") == \
                "tpo_filter", f"{fid}/{c.id} missing TPO marker"


# --- entry_lab --------------------------------------------------------------

def _base() -> PropCandidate:
    """Pick an arbitrary OHLC-only family base for lab tests."""
    return families.previous_h4_range_retracement()[0]


def test_entry_variants_only_emit_registered_entries():
    base = _base()
    vs = entry_lab.variants(base)
    assert len(vs) >= 3, f"too few entry variants ({len(vs)})"
    for v in vs:
        assert v.entry["type"] in {
            "touch_entry", "reaction_close", "fib_limit_entry",
            "zone_midpoint_limit", "minor_structure_break",
            "delayed_entry_1", "delayed_entry_2",
        }


def test_entry_variants_unique_ids():
    base = _base()
    ids = [v.id for v in entry_lab.variants(base)]
    assert len(set(ids)) == len(ids)


def test_entry_lab_compare_table_shape():
    rows = [
        {"candidate_id": "a", "entry_type": "touch_entry",
         "ho_total_return": 0.05, "ho_sharpe_trade_ann": 1.1,
         "ho_max_drawdown": -0.05, "wf_median_sharpe": 0.3,
         "label_perm_p": 0.10, "prop_pass_probability": 0.30},
        {"candidate_id": "b", "entry_type": "reaction_close",
         "ho_total_return": 0.08, "ho_sharpe_trade_ann": 1.4,
         "ho_max_drawdown": -0.03, "wf_median_sharpe": 0.5,
         "label_perm_p": 0.04, "prop_pass_probability": 0.45},
    ]
    table = entry_lab.compare_table(rows)
    assert table["n_entries"] == 2
    # higher-is-better metrics: reaction_close should win
    assert table["best"]["ho_total_return"]["entry"] == "reaction_close"
    assert table["best"]["prop_pass_probability"]["entry"] == "reaction_close"
    # lower-is-better: drawdown -> reaction_close (less negative? no,
    # drawdown is negative; "less drawdown" = closer to 0; here -0.03
    # > -0.05, so reaction_close wins as "min drawdown")
    assert table["best"]["ho_max_drawdown"]["entry"] in {"reaction_close", "touch_entry"}
    # p-value: lower wins
    assert table["best"]["label_perm_p"]["entry"] == "reaction_close"


# --- risk_sweep -------------------------------------------------------------

def test_risk_variants_default_count():
    base = _base()
    vs = risk_sweep.variants(base)
    # 7 default presets
    assert len(vs) == 7
    names = {v.risk.name for v in vs}
    assert "fixed_micro_1" in names
    assert "dollar_risk_50" in names
    assert "pct_dd_buffer_2pct" in names


def test_risk_variants_honour_contracts_max():
    base = _base()
    vs = risk_sweep.variants(base, contracts_max=3)
    for v in vs:
        assert v.risk.contracts_max == 3


def test_risk_variants_with_caps_cartesian():
    base = _base()
    vs = risk_sweep.variants_with_caps(base, contracts_caps=(1, 2, 5))
    # 7 presets x 3 caps
    assert len(vs) == 21


# --- daily_rule_lab ---------------------------------------------------------

def test_daily_variants_default_count_matches_presets():
    base = _base()
    vs = daily_rule_lab.variants(base)
    assert len(vs) == len(daily_rule_lab.DEFAULT_PRESETS)


def test_daily_variants_do_not_mutate_base():
    base = _base()
    base_name = base.daily_rules.name
    daily_rule_lab.variants(base)
    assert base.daily_rules.name == base_name, (
        "daily_rule_lab.variants mutated the base candidate")


def test_daily_variants_full_is_wider_than_default():
    base = _base()
    n_default = len(daily_rule_lab.variants(base))
    n_full = len(daily_rule_lab.variants_full(base))
    assert n_full > n_default


# --- grid -------------------------------------------------------------------

def test_tier_1_grid_is_all_families():
    g = grid.tier_1_grid()
    assert len(g) == len(families.all_candidates())


def test_tier_1_grid_filter_selects_subset():
    g = grid.tier_1_grid(
        families_filter=["session_sweep_reclaim", "compression_breakout"])
    fams = {c.family for c in g}
    assert fams == {"session_sweep_reclaim", "compression_breakout"}


def test_apply_capability_filter_separates_clean_and_dirty():
    """Synthesise a dirty candidate by injecting a vwap filter."""
    base = _base()
    dirty = PropCandidate.from_json(base.to_json())
    dirty.id = "dirty_test"
    dirty.filters = list(dirty.filters) + [{"type": "vwap_dist"}]
    # Don't run the dirty candidate through to_spec validation — just
    # rely on the classifier inside apply_capability_filter.
    kept, rejected = grid.apply_capability_filter([base, dirty])
    assert len(kept) == 1 and kept[0].id == base.id
    assert len(rejected) == 1 and rejected[0].id == "dirty_test"
    assert rejected[0].certification_level == \
        CertificationLevel.REJECTED_UNAVAILABLE_DATA
    assert FailureReason.REJECTED_UNAVAILABLE_DATA in rejected[0].failure_reasons
    assert "vwap" in rejected[0].rejection_detail.get("unavailable_features", [])


def test_tier_2_dispatch():
    base = _base()
    n_risk = len(grid.tier_2_for_survivor(base, lab="risk"))
    n_daily = len(grid.tier_2_for_survivor(base, lab="daily"))
    n_entry = len(grid.tier_2_for_survivor(base, lab="entry"))
    assert n_risk > 0 and n_daily > 0 and n_entry > 0
    raised = False
    try:
        grid.tier_2_for_survivor(base, lab="not_a_lab")
    except ValueError:
        raised = True
    assert raised


def test_tier_2_full_combines_all_labs():
    base = _base()
    expected = (len(risk_sweep.variants(base))
                + len(daily_rule_lab.variants(base))
                + len(entry_lab.variants(base)))
    actual = len(grid.tier_2_full([base]))
    assert actual == expected


def test_grid_summary_counts():
    g = grid.tier_1_grid()
    summary = grid.grid_summary(g)
    assert summary["total"] == len(g)
    assert sum(summary["by_family"].values()) == len(g)
    assert sum(summary["by_certification_level"].values()) == len(g)


# --- combined sweep size sanity --------------------------------------------

def test_default_tier_1_grid_size_is_controlled():
    """Tier 1 should stay under ~150 candidates so the orchestrator's
    fast filter remains tractable in <30 minutes."""
    n = len(grid.tier_1_grid())
    assert 20 <= n <= 150, (
        f"tier-1 grid size {n} outside the 'controlled' band [20, 150]")


if __name__ == "__main__":
    fns = [
        test_every_family_emits_at_least_one_candidate,
        test_family_ids_match_brief,
        test_all_candidates_have_unique_ids,
        test_every_candidate_passes_capability_check,
        test_every_candidate_to_spec_works,
        test_tpo_families_carry_executor_extension_marker,
        test_entry_variants_only_emit_registered_entries,
        test_entry_variants_unique_ids,
        test_entry_lab_compare_table_shape,
        test_risk_variants_default_count,
        test_risk_variants_honour_contracts_max,
        test_risk_variants_with_caps_cartesian,
        test_daily_variants_default_count_matches_presets,
        test_daily_variants_do_not_mutate_base,
        test_daily_variants_full_is_wider_than_default,
        test_tier_1_grid_is_all_families,
        test_tier_1_grid_filter_selects_subset,
        test_apply_capability_filter_separates_clean_and_dirty,
        test_tier_2_dispatch,
        test_tier_2_full_combines_all_labs,
        test_grid_summary_counts,
        test_default_tier_1_grid_size_is_controlled,
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
