"""End-to-end demo: backtest → RL train → RL evaluate.

Run:
    python demo.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    from prop_backtest.contracts.specs import get_contract
    from prop_backtest.data.loader import DataLoader
    from prop_backtest.engine.backtest import BacktestEngine
    from prop_backtest.firms import get_firm
    from prop_backtest.reporting.report import print_report, save_report_csv
    from prop_backtest.strategy.examples.sma_crossover import SMACrossover
    from prop_backtest.rl.trainer import train as rl_train
    from prop_backtest.rl.rewards import RewardShaper
    from prop_backtest.strategy.rl_strategy import RLStrategy

    ES   = get_contract("ES")
    FIRM = get_firm("topstep")
    TIER = "100K"

    # ── 1. Load synthetic data ─────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1 — LOAD DATA")
    print("=" * 60)

    loader     = DataLoader(ES)
    train_bars = loader.load("2023-01-01", "2025-01-01", local_csv_path="/tmp/ES_train.csv")
    test_bars  = loader.load("2025-01-01", "2025-07-01", local_csv_path="/tmp/ES_test.csv")

    print(f"Train: {len(train_bars)} bars  ({train_bars[0].date} → {train_bars[-1].date})")
    print(f"Test:  {len(test_bars)} bars   ({test_bars[0].date} → {test_bars[-1].date})")

    # ── 2. SMA crossover backtest ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 — SMA CROSSOVER BACKTEST  [fast=3, slow=10]")
    print("=" * 60)

    sma_result = BacktestEngine(
        strategy=SMACrossover(fast=3, slow=10, contracts=1, risk_aware=True),
        firm_rules=FIRM, tier_name=TIER, contract=ES,
        commission_per_rt=4.50, slippage_ticks=0,
    ).run(train_bars)

    print_report(sma_result)
    save_report_csv(sma_result, "/tmp/sma_results")

    # ── 3. RL training ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — PPO RL AGENT TRAINING  [50k steps]")
    print("=" * 60)

    model = rl_train(
        firm_rules=FIRM, tier_name=TIER, contract=ES,
        bars=train_bars,
        total_timesteps=50_000,
        model_path="/tmp/ppo_es_topstep",
        n_envs=2, window=20,
        commission_per_rt=4.50, slippage_ticks=0,
        reward_shaper=RewardShaper(alpha=0.5, pass_bonus=2.0, fail_penalty=-1.0),
        eval_freq=10_000, verbose=1,
    )

    # ── 4. Evaluate RL agent on held-out test data ─────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — RL AGENT EVALUATION  [out-of-sample test data]")
    print("=" * 60)

    rl_result = BacktestEngine(
        strategy=RLStrategy(model_path="/tmp/ppo_es_topstep", window=20),
        firm_rules=FIRM, tier_name=TIER, contract=ES,
        commission_per_rt=4.50, slippage_ticks=0,
    ).run(test_bars)

    print_report(rl_result)
    save_report_csv(rl_result, "/tmp/rl_results")

    # ── 5. Comparison summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Net PnL':>10} {'Trades':>8} {'Win %':>8} {'Verdict':>16}")
    print("-" * 64)
    for name, r in [("SMA(3/10)", sma_result), ("PPO RL Agent", rl_result)]:
        pnl  = r.final_realized_balance - r.starting_balance
        tr   = r.stats.get("total_trades", 0)
        wr   = r.stats.get("win_rate", 0) * 100
        verd = "PASS ✓" if r.passed else f"FAIL ({r.failure_reason})"
        print(f"{name:<20} {pnl:>+10,.2f} {tr:>8} {wr:>7.1f}% {verd:>16}")
    print("=" * 64)


if __name__ == "__main__":
    main()
