#!/usr/bin/env python3
"""Optimize pass rate for 22-day (30 calendar day) challenge windows."""

import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from src.data.feed import CSVDataFeed
from src.backtest.challenge_sim import ChallengeSim
from src.core.config import load_global_config
from src.strategy.breakout import BreakoutStrategy
from src.strategy.level_sweep import LevelSweepStrategy
from src.strategy.supply_demand import SupplyDemandStrategy
from src.data.models import ChallengeStatus

feed = CSVDataFeed("data")
es = feed.get_bars("ES", "5min").loc["2015":"2019"]
gc = feed.get_bars("GC", "5min").loc["2015":"2019"]
nq = feed.get_bars("NQ", "5min").loc["2015":"2019"]
print(f"ES: {len(es)} bars, GC: {len(gc)} bars, NQ: {len(nq)} bars", flush=True)


def make_strategies(rr=3.0, sweep_params=None, sd_params=None, breakout_params=None):
    bp = {"opening_range_minutes": 60, "stop_range_pct": 0.5, "take_profit_rr": rr,
          "min_range_points": 0.5, "max_range_points": 20.0, "max_bars_after_or": 18}
    if breakout_params:
        bp.update(breakout_params)

    sp = {"sweep_buffer_ticks": 2, "rejection_wick_pct": 0.40, "stop_buffer_ticks": 4,
          "take_profit_rr": rr, "max_sweep_ticks": 40, "use_pdh_pdl": True,
          "use_swing_levels": True, "swing_lookback_bars": 48}
    if sweep_params:
        sp.update(sweep_params)

    sdp = {"zone_atr_mult": 2.0, "zone_body_pct": 0.75, "stop_buffer_ticks": 12,
           "take_profit_rr": rr, "max_zone_age_bars": 240, "rejection_wick_pct": 0.55}
    if sd_params:
        sdp.update(sd_params)

    return {
        "breakout": BreakoutStrategy(bp),
        "level_sweep": LevelSweepStrategy(sp),
        "supply_demand": SupplyDemandStrategy(sdp),
    }


def run_test(instrument_data, risk_pct=15.0, rr=3.0, label="", **kwargs):
    config = load_global_config("config")
    config.risk.max_risk_per_trade_pct = risk_pct
    strategies = make_strategies(rr=rr, **kwargs)
    sim = ChallengeSim(config, strategies)
    t0 = time.time()
    results = sim.run_multiple_multi(instrument_data, n_runs=9999, window_days=22)
    dt = time.time() - t0
    n = len(results)
    if n == 0:
        print(f"{label}: NO RESULTS", flush=True)
        return
    passed = sum(1 for r in results if r.passed)
    blown = sum(1 for r in results if r.backtest_result.status == ChallengeStatus.BLOWN)
    avg_trades = sum(r.backtest_result.metrics.get("total_trades", 0) for r in results) / max(n, 1)
    avg_profit = sum(r.profit for r in results) / max(n, 1)
    print(f"{label}: {passed}/{n} = {passed/n:.1%} pass, {blown/n:.1%} blown, "
          f"{avg_trades:.0f} tr/w, ${avg_profit:+,.0f}, {dt:.0f}s", flush=True)


# Test matrix
print("\n=== Risk per trade tests (ES+GC) ===", flush=True)
run_test({"ES": es, "GC": gc}, risk_pct=15.0, rr=3.0, label="r15 RR3")
run_test({"ES": es, "GC": gc}, risk_pct=20.0, rr=3.0, label="r20 RR3")
run_test({"ES": es, "GC": gc}, risk_pct=25.0, rr=3.0, label="r25 RR3")

print("\n=== Triple instrument tests (ES+GC+NQ) ===", flush=True)
run_test({"ES": es, "GC": gc, "NQ": nq}, risk_pct=15.0, rr=3.0, label="3inst r15 RR3")
run_test({"ES": es, "GC": gc, "NQ": nq}, risk_pct=20.0, rr=3.0, label="3inst r20 RR3")

print("\n=== R:R tests (ES+GC, r15) ===", flush=True)
run_test({"ES": es, "GC": gc}, risk_pct=15.0, rr=4.0, label="r15 RR4")
run_test({"ES": es, "GC": gc}, risk_pct=15.0, rr=2.5, label="r15 RR2.5")

print("\nDone.", flush=True)
