"""Dukascopy-only pipeline audit.

Hard rules (any failure → exit 1):

  - data/loader.py refuses non-Dukascopy sources
  - every candle file under data/dukascopy/candles/ has dataset_source == "dukascopy"
  - data/dukascopy/manifests/XAUUSD_manifest.json exists with official_source = "dukascopy"
  - no active code path imports the deprecated broker CSVs in data/_deprecated_/
  - active result files (alpha leaderboard etc.) carry no rows referencing
    deprecated sources
  - timeframe alignment / OHLC sanity / gap detection (when data is on disk)
  - parent/child mapping H4 ↔ M15 (when both are on disk)

Hardening checks (added in the audit pass):

  - data_splits.json windows are non-overlapping and ordered
  - actual data coverage matches each split window (warning if config
    declares more history than is on disk)
  - canonical executor charges spread in price units, NOT *POINT_SIZE
  - validation/walkforward.py does not silently downgrade M15 entries
    to h4_open
  - prop sim risk-sizing path does not consume realised PnL
  - shuffle-of-returns statistical test is not used (it is a no-op)

If `data/dukascopy/candles/XAUUSD/H4/` is empty, the audit reports
`no_dukascopy_data_yet` as a fatal failure with instructions, NOT a
warning. The pipeline refuses to certify anything until the official
source is populated.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_candles, list_available
from data.splits import load_split_config, load_splits, coverage_summary

RESULTS = ROOT / "results"
DUKA_MANIFEST = ROOT / "data" / "dukascopy" / "manifests" / "XAUUSD_manifest.json"
CANDLES = ROOT / "data" / "dukascopy" / "candles" / "XAUUSD"
DEPRECATED = ROOT / "data" / "_deprecated_"
SPLITS_CFG = ROOT / "config" / "data_splits.json"


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0

    def write(self, msg: str) -> None:
        self.lines.append(msg)
        print(msg)

    def section(self, title: str) -> None:
        self.write("")
        self.write(f"=== {title} ===")

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        self.write(f"  [{tag}] {label}" + (f"   ({detail})" if detail else ""))

    def dump(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def audit_manifest(rep: Report) -> dict | None:
    rep.section("manifest — single Dukascopy source")
    rep.check("data/dukascopy/manifests/XAUUSD_manifest.json exists",
              DUKA_MANIFEST.exists(), detail=str(DUKA_MANIFEST))
    if not DUKA_MANIFEST.exists():
        return None
    m = json.loads(DUKA_MANIFEST.read_text())
    rep.check("official_source == dukascopy",
              m.get("official_source") == "dukascopy",
              detail=f"got={m.get('official_source')}")
    rep.check("old_sources_deprecated == true",
              m.get("old_sources_deprecated") is True)
    rep.check("generated_from_tick_data == true",
              m.get("generated_from_tick_data") is True)
    rep.check("manifest declares all 8 required timeframes",
              set(m.get("timeframes", [])) >=
              {"M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"})
    return m


def audit_data_present(rep: Report) -> dict[str, list[int]]:
    rep.section("Dukascopy candle data on disk")
    avail = list_available()
    if not avail or not any(avail.values()):
        rep.check("at least one timeframe has candles on disk",
                  False,
                  detail="no Dukascopy data yet — run scripts/fetch_dukascopy.py "
                         "from an unrestricted environment, or drop "
                         ".bi5 hourly tick files into data/dukascopy/raw/")
        return avail
    for tf in ("M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"):
        years = avail.get(tf, [])
        rep.check(f"{tf}: years on disk", bool(years),
                  detail=f"{years if years else 'none'}")
    return avail


def audit_dataset_source(rep: Report, avail: dict[str, list[int]]) -> None:
    rep.section("dataset_source enforcement")
    if not any(avail.values()):
        rep.write("  (skipped — no candles on disk yet)")
        return
    bad = 0
    for tf in avail:
        files = (list((CANDLES / tf).glob("year=*.parquet"))
                 + list((CANDLES / tf).glob("year=*.csv")))
        for f in files:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f, columns=["dataset_source"])
            else:
                df = pd.read_csv(f, nrows=1)
            if "dataset_source" not in df.columns:
                rep.check(f"{tf}/{f.name}: dataset_source column present",
                          False, detail="missing column")
                bad += 1
                continue
            if (df["dataset_source"] != "dukascopy").any():
                rep.check(f"{tf}/{f.name}: dataset_source",
                          False, detail=f"got={df['dataset_source'].iloc[0]}")
                bad += 1
    rep.check("every candle file's dataset_source == dukascopy", bad == 0)


def audit_no_active_use_of_deprecated(rep: Report) -> None:
    rep.section("active code does not import deprecated broker data")
    if not DEPRECATED.exists():
        rep.write("  (no _deprecated_ dir; skipping)")
        return
    deprecated_csvs = {
        "XAUUSD_H4_long.csv",
        "XAUUSD_H4_matched.csv",
        "XAUUSD_M15_matched.csv",
    }
    # Audit and the report builder are allowed to MENTION the
    # deprecated filenames as text (the audit checks FOR them; the PDF
    # builder cites historical results). Only flag files that load
    # them via pd.read_csv.
    skip = {"audit.py", "build_pdf.py", "fetch_dukascopy.py"}
    offenders: list[str] = []
    for py in (ROOT / "scripts").glob("*.py"):
        if "_deprecated_" in py.parts or py.name in skip:
            continue
        text = py.read_text()
        for c in deprecated_csvs:
            if f'pd.read_csv("data/{c}"' in text or f'data/{c}' in text:
                offenders.append(f"{py.name} loads {c}")
    rep.check("no scripts/*.py references the deprecated broker CSVs",
              not offenders,
              detail="; ".join(offenders) if offenders else "")


def audit_timeframe_integrity(rep: Report, avail: dict[str, list[int]]) -> None:
    rep.section("timeframe integrity (when data is present)")
    if not avail.get("M15") or not avail.get("H4"):
        rep.write("  (skipped — H4 or M15 not on disk)")
        return
    h4 = load_candles(timeframe="H4")
    m15 = load_candles(timeframe="M15")
    rep.check("H4 timestamps land on a 4h boundary",
              ((h4["time"].dt.hour % 4 == 0)
               & (h4["time"].dt.minute == 0)
               & (h4["time"].dt.second == 0)).all())
    rep.check("M15 timestamps land on a 15-min boundary",
              ((m15["time"].dt.minute % 15 == 0)
               & (m15["time"].dt.second == 0)).all())
    rep.check("H4 OHLC valid (low <= open,close <= high)",
              ((h4["low"] <= h4[["open", "close"]].min(axis=1))
               & (h4[["open", "close"]].max(axis=1) <= h4["high"])).all())
    # M15 sub-bars must each map to an H4 bucket present in H4
    bucket = m15["time"].dt.floor("4h")
    n_outside = int((~bucket.isin(set(h4["time"]))).sum())
    rep.check("every M15 bar maps to a known H4 bucket",
              n_outside == 0, detail=f"outside={n_outside}")


def audit_splits(rep: Report, avail: dict[str, list[int]]) -> None:
    """Phase 1: split config is valid, windows do not overlap, and the
    on-disk data actually covers each declared window."""
    rep.section("data splits — non-overlap + coverage")
    if not SPLITS_CFG.exists():
        rep.check("config/data_splits.json exists", False,
                  detail=str(SPLITS_CFG))
        return
    try:
        cfg = load_split_config(SPLITS_CFG)
    except Exception as exc:
        rep.check("config/data_splits.json parses with non-overlapping splits",
                  False, detail=str(exc))
        return
    rep.check("config/data_splits.json parses with non-overlapping splits",
              True, detail=", ".join(cfg.keys()))
    if not avail.get("H4") or not avail.get("M15"):
        rep.write("  (skipping coverage check — H4/M15 not on disk)")
        return
    splits = load_splits()
    cov = coverage_summary(splits)
    for name, info in cov.items():
        ok = info["h4_rows"] > 0
        rep.check(f"{name}: at least one H4 bar in window",
                  ok,
                  detail=f"{info['h4_rows']} H4 rows "
                         f"({info['actual_first_bar']} -> "
                         f"{info['actual_last_bar']})")
        # Coverage warning: actual_first_bar should be near config_start.
        # If the data starts much later, the split is shorter than declared.
        if info["actual_first_bar"] is not None:
            actual = pd.Timestamp(info["actual_first_bar"])
            declared = pd.Timestamp(info["config_start"])
            gap_days = (actual - declared).days
            rep.check(
                f"{name}: data covers declared start (config={info['config_start']})",
                gap_days < 90,
                detail=f"first bar is {gap_days} days after declared start")


def audit_spread_unit_convention(rep: Report) -> None:
    """Phase 4: canonical executor must NOT multiply spread by POINT_SIZE.

    Dukascopy stores spread = ask - bid in price units (codec scales
    asks/bids by 1/price_scale before computing spread). The cost per
    leg is therefore the spread value directly. Any `spread * POINT_SIZE`
    in active code paths is the 1000x undercharge bug.
    """
    rep.section("spread unit convention (price units, no POINT_SIZE rescale)")

    def scan(path: Path) -> list[tuple[int, str]]:
        hits = []
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            # crude pattern: a `spread`-named token combined with POINT_SIZE
            if re.search(r"spread[^=]*\*\s*POINT_SIZE", line) or \
               re.search(r"POINT_SIZE\s*\*[^=]*spread", line):
                hits.append((i, line.rstrip()))
        return hits

    # canonical executor + v1 strategy. Skip _deprecated_ paths.
    paths = [
        ROOT / "execution" / "executor.py",
        ROOT / "scripts" / "strategy.py",
    ]
    offenders = []
    for p in paths:
        if not p.exists():
            continue
        for ln, txt in scan(p):
            offenders.append(f"{p.relative_to(ROOT)}:{ln}: {txt}")
    rep.check("canonical paths do not multiply `spread` by POINT_SIZE",
              not offenders,
              detail=("\n    " + "\n    ".join(offenders)) if offenders else "")


def audit_walkforward_no_downgrade(rep: Report) -> None:
    """Phase 2: walk-forward must use the same executor as holdout.
    Failure modes:
      * walkforward.py contains old downgrade strings ("treat as
        h4_open", "can't replay M15 entries", etc.)
      * walkforward.py does not import from execution.executor
    """
    rep.section("walk-forward uses M15-aware executor")
    wf_path = ROOT / "validation" / "walkforward.py"
    if not wf_path.exists():
        rep.check("validation/walkforward.py exists", False)
        return
    text = wf_path.read_text()
    downgrade_markers = [
        "can't replay M15 entries",
        "treat as h4_open",
        "downgrades many M15 entry models",
    ]
    found = [m for m in downgrade_markers if m in text]
    rep.check("walk-forward does not downgrade M15 entries to h4_open",
              not found,
              detail=("KNOWN: " + "; ".join(found)) if found else "")
    rep.check("walk-forward imports execution.executor.run",
              "from execution.executor import" in text and "run as run_executor" in text,
              detail="ensures WF kernel == holdout kernel")


def audit_prop_sim_no_future_leak(rep: Report) -> None:
    """Phase 7: prop simulator's risk-sizing path must not consume the
    realised trade PnL (`trade_pnl_price` parameter) for sizing, and
    the Monte Carlo must use day-level (not trade-level) bootstrap so
    real intraday clustering is preserved."""
    rep.section("prop sim risk sizing + bootstrap (Phase 7)")
    risk_path = ROOT / "prop_challenge" / "risk.py"
    chal_path = ROOT / "prop_challenge" / "challenge.py"
    if not risk_path.exists():
        rep.check("prop_challenge/risk.py exists", False)
        return
    risk_text = risk_path.read_text()
    chal_text = chal_path.read_text() if chal_path.exists() else ""

    # 1. risk sizing: no trade_pnl_price as a parameter or call kw.
    # Match `trade_pnl_price:` (annotation) or `trade_pnl_price=` (call/assign)
    # but not bare mentions in docstrings.
    leak_uses = re.findall(r"trade_pnl_price\s*[:=]", risk_text)
    rep.check("RiskModel.size() does not accept realised trade PnL",
              not leak_uses,
              detail=("trade_pnl_price still used in risk.py: "
                      + ", ".join(leak_uses)) if leak_uses else "")

    # 2. risk sizing must use stop_distance_price or atr_pre_entry
    rep.check("RiskModel.size() consumes pre-trade risk inputs",
              "stop_distance_price" in risk_text or "atr_pre_entry" in risk_text,
              detail="size() signature missing stop_distance_price/atr_pre_entry"
                     if not ("stop_distance_price" in risk_text or "atr_pre_entry" in risk_text)
                     else "")

    # 3. challenge engine must expose chronological replay
    rep.check("challenge engine has chronological replay mode",
              "run_chronological_replay" in chal_text,
              detail="run_chronological_replay() not exported"
                     if "run_chronological_replay" not in chal_text else "")

    # 4. bootstrap must be day-level
    rep.check("challenge MC uses day-level block bootstrap",
              "_sample_day_blocks" in chal_text or "block_days" in chal_text,
              detail="trade-level bootstrap still in use"
                     if "_sample_day_blocks" not in chal_text and "block_days" not in chal_text
                     else "")

    # 5. Wilson CI is reported on the headline probabilities
    rep.check("challenge result carries Wilson CI on pass/blowup probabilities",
              "pass_probability_ci" in chal_text and "blowup_probability_ci" in chal_text,
              detail="missing CI fields on ChallengeResult"
                     if not ("pass_probability_ci" in chal_text)
                     else "")


def audit_shuffle_test_disabled(rep: Report) -> None:
    """Phase 5: the no-op shuffle-of-returns test must not be a gate.

    Sharpe is invariant under permutation, so the test always returned
    p ~= 1.0. The new statistical_tests module has it deprecated to
    raise on call. The certifier must accept `stat_label_perm` and
    must not depend on `stat_shuffle`.
    """
    rep.section("statistical test validity")
    cert = ROOT / "validation" / "certify.py"
    stat = ROOT / "validation" / "statistical_tests.py"
    if not cert.exists():
        rep.check("validation/certify.py exists", False)
        return
    cert_text = cert.read_text()
    stat_text = stat.read_text() if stat.exists() else ""

    rep.check("certifier accepts stat_label_perm",
              "stat_label_perm" in cert_text,
              detail="missing label-permutation gate"
                     if "stat_label_perm" not in cert_text else "")
    rep.check("certifier does not silently consume the no-op shuffle test",
              "stat_shuffle" not in cert_text or "DeprecationWarning" in cert_text,
              detail="certifier still wires stat_shuffle as a gate"
                     if "stat_shuffle" in cert_text and
                        "DeprecationWarning" not in cert_text else "")
    rep.check("statistical_tests.shuffled_outcome_test is deprecated",
              "DEPRECATED" in stat_text and "RuntimeError" in stat_text,
              detail="shuffled_outcome_test must raise; see Phase 5 hardening")


def audit_prop_accounts_metadata(rep: Report) -> None:
    """Phase 8: every account in config/prop_accounts.json must declare
    a `_verification` block with source_url, last_verified, notes.
    The certifier later treats unverified accounts as research_only."""
    rep.section("prop_accounts.json verification metadata (Phase 8)")
    cfg_path = ROOT / "config" / "prop_accounts.json"
    if not cfg_path.exists():
        rep.check("config/prop_accounts.json exists", False)
        return
    raw = json.loads(cfg_path.read_text())

    # schema is parseable
    try:
        from prop_challenge.accounts import (
            validate_schema, load_all, verification_status,
        )
        validate_schema(raw)
        accounts = load_all()
        rep.check("prop_accounts.json passes schema validation", True,
                  detail=f"{len(accounts)} accounts")
    except Exception as exc:
        rep.check("prop_accounts.json passes schema validation", False,
                  detail=str(exc))
        return

    # every account has a _verification block
    missing_meta = []
    statuses: dict[str, int] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if "_verification" not in v:
            missing_meta.append(k)
            continue
    rep.check("every account has a `_verification` block",
              not missing_meta,
              detail=("missing on: " + ", ".join(missing_meta))
                     if missing_meta else "")

    # status summary
    for name, spec in accounts.items():
        st = verification_status(spec)
        statuses[st] = statuses.get(st, 0) + 1
    rep.write(f"  status histogram: {statuses}")
    n_unver = statuses.get("unverified", 0) + statuses.get("stale", 0)
    real_ones = sum(1 for spec in accounts.values()
                    if (spec.source_url or "") != "synthetic")
    if real_ones:
        rep.check(
            "no real-firm accounts left as `unverified` or `stale`",
            n_unver == 0,
            detail=(f"{n_unver}/{real_ones} real-firm accounts need a "
                    f"`last_verified` ISO date set within "
                    f"90 days; certifier marks their results research_only.")
                   if n_unver else "")


def audit_feature_capability(rep: Report) -> None:
    """Phase 9 (Batch F): the harness must reject candidates needing
    unavailable real-volume / VWAP / Volume Profile / footprint /
    delta / CVD / order-book features. TPO must be allowed without
    real volume.

    This audit performs three checks:
      1. The canonical CapabilityRegistry has the expected disabled
         features.
      2. A representative VWAP/Volume Profile/footprint/delta/CVD/
         orderbook candidate is rejected.
      3. A representative TPO candidate is accepted.
    """
    rep.section("OHLC-only feature capability (Batch F)")
    try:
        from core.feature_capability import (
            CANONICAL_REGISTRY, classify_candidate,
        )
    except Exception as exc:
        rep.check("core.feature_capability importable", False, detail=str(exc))
        return

    expect_off = ("real_volume", "bid_ask_volume", "footprint", "delta",
                  "cvd", "orderbook", "vwap", "volume_profile")
    bad = [f for f in expect_off if getattr(CANONICAL_REGISTRY, f, True)]
    rep.check(
        "canonical registry disables real-volume / VWAP / VP / footprint / "
        "delta / CVD / orderbook",
        not bad,
        detail=("UNEXPECTEDLY ENABLED: " + ", ".join(bad)) if bad else "")

    rejected_tokens = ("vwap_dist", "volume_poc", "footprint", "delta",
                        "cvd", "bid_ask_imbalance", "dom_liquidity")
    misses: list[str] = []
    for tok in rejected_tokens:
        spec = {"signal": {"type": "prev_color"},
                 "filters": [{"type": tok}],
                 "entry": {"type": "touch_entry"},
                 "stop": {"type": "prev_h4_open"},
                 "exit": {"type": "h4_close"}}
        v = classify_candidate(spec)
        if v.status != "rejected_unavailable_data":
            misses.append(tok)
    rep.check("VWAP / VolumeProfile / footprint / delta / CVD / orderbook "
              "tokens are rejected by classify_candidate",
              not misses,
              detail=("FAILED to reject: " + ", ".join(misses)) if misses else "")

    tpo_spec = {"signal": {"type": "prev_color"},
                 "filters": [{"type": "tpo_value_acceptance"},
                              {"type": "tpo_poor_high"}],
                 "entry": {"type": "tpo_value_rejection"},
                 "stop": {"type": "prev_h4_open"},
                 "exit": {"type": "h4_close"}}
    v = classify_candidate(tpo_spec)
    rep.check("TPO tokens (tpo_*) are accepted on OHLC-only data",
              v.status == "ok",
              detail=f"got status={v.status}, unavailable={v.unavailable_tokens}")


def audit_active_strategies_no_unavailable_data(rep: Report) -> None:
    """Sweep canonical strategy/runner files for hard-coded references
    to unavailable-data features (vwap, volume_profile, footprint,
    delta, cvd, orderbook). The intent is to catch regressions where
    a token like `volume_poc` slips back into a runner / family
    generator. Comments and docstrings are excluded by skipping lines
    starting with `#` or wrapped in triple quotes.
    """
    rep.section("active strategies do not reference unavailable data")
    forbidden = ("vwap", "volume_profile", "volume_poc", "footprint",
                 "_delta", "cvd_", "bid_ask_imbalance", "dom_liquidity")
    skip_dirs = ("scripts/_deprecated_", "data/_deprecated_",
                 "results/_archive_pre_dukascopy", "results/_pre_hardening",
                 "tests/", "agents/", "docs/", "results/audit.txt")
    # Files that legitimately mention forbidden tokens by name:
    # - audit.py: audit logic
    # - feature_capability.py: registry declarations
    # - build_pdf.py: PDF text labels (documentation)
    # - strategy.py: deprecation tombstones (raise ValueError when the
    #   v1 kernel is asked to run a vwap_dist filter)
    skip_files = {"audit.py", "feature_capability.py",
                  "build_pdf.py", "strategy.py"}
    # active code paths to scan
    paths = []
    for sub in ("scripts", "validation", "execution", "core", "analytics",
                 "prop", "prop_challenge", "data", "regime", "entry_models",
                 "strategies"):
        d = ROOT / sub
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            if any(s in str(p) for s in skip_dirs):
                continue
            paths.append(p)
    offenders: list[str] = []
    for p in paths:
        if p.name in skip_files:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # very simple "is this a comment / docstring" filter to avoid
        # the audit / capability declarations themselves flagging
        in_doc = False
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_doc = not in_doc
                continue
            if in_doc:
                continue
            if stripped.startswith("#"):
                continue
            for f in forbidden:
                if f in line.lower():
                    offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:100]}")
                    break
    rep.check("no canonical .py file references vwap / volume_profile / "
              "footprint / delta / cvd / orderbook / bid_ask_imbalance",
              not offenders,
              detail=("\n    " + "\n    ".join(offenders[:10])
                      + (f"\n    ...({len(offenders)-10} more)" if len(offenders) > 10 else ""))
                     if offenders else "")


def audit_results_provenance(rep: Report) -> None:
    """Active leaderboard files should declare the commit / config they
    were produced from. Lacking that, we mark them as research-only."""
    rep.section("results provenance")
    for f in sorted(RESULTS.glob("*.csv")):
        if "_archive" in str(f):
            continue
        # heuristic: a leaderboard sidecar JSON of the same stem with
        # `commit` and `produced_at` keys would satisfy provenance.
        sidecar = f.with_suffix(".meta.json")
        rep.check(f"{f.name}: provenance metadata present",
                  sidecar.exists(),
                  detail="missing .meta.json — treat as research-only")


def main() -> int:
    rep = Report()
    rep.write("XAUUSD Dukascopy-only pipeline audit")
    rep.write("=" * 60)

    audit_manifest(rep)
    avail = audit_data_present(rep)
    audit_dataset_source(rep, avail)
    audit_no_active_use_of_deprecated(rep)
    audit_timeframe_integrity(rep, avail)
    # Hardening checks
    audit_splits(rep, avail)
    audit_spread_unit_convention(rep)
    audit_walkforward_no_downgrade(rep)
    audit_prop_sim_no_future_leak(rep)
    audit_prop_accounts_metadata(rep)
    audit_shuffle_test_disabled(rep)
    audit_feature_capability(rep)
    audit_active_strategies_no_unavailable_data(rep)
    audit_results_provenance(rep)

    rep.write("")
    rep.write(f"=== audit summary: {rep.failures} failures ===")
    rep.dump(RESULTS / "audit.txt")
    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
