"""Strict certification engine.

A strategy is VALID ONLY IF all of these hold:

  WF folds                  >= 20
  WF median Sharpe          >  0.5
  WF positive folds         >= 60%
  holdout total return      >  0
  profit factor             >  1.2
  max drawdown              >= -0.20    (i.e. no worse than 20% on holdout)
  no single trade > 20%     of total profit
  passes statistical sig    (shuffled p < 0.05)
  passes random baseline    (real Sharpe > 95th pct random)
  passes execution stress   (still profitable under stress execution)
  yearly consistency        passes (>=3 positive years)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CertResult:
    certified: bool
    failures: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def certify(*, wf: dict, holdout_metrics: dict, holdout_stress: dict,
            stat_shuffle: dict, stat_random: dict, yearly: dict,
            dataset_source: str = "unknown",
            min_folds: int = 20, min_median_sharpe: float = 0.5,
            min_positive_folds: float = 0.60, min_profit_factor: float = 1.2,
            max_drawdown_floor: float = -0.20, max_biggest_share: float = 0.20,
            ) -> CertResult:
    fails: list[str] = []
    detail: dict = {"dataset_source": dataset_source}

    # 0. Official-source gate (added in Dukascopy migration).
    if dataset_source != "dukascopy":
        fails.append(f"non_official_data_source (got {dataset_source!r}); "
                     "active certifier accepts dataset_source='dukascopy' only")
        return CertResult(certified=False, failures=fails, detail=detail)

    # 1. WF folds
    n_folds = wf.get("folds", 0)
    detail["wf_folds"] = n_folds
    if n_folds < min_folds:
        fails.append(f"wf_folds<{min_folds} (got {n_folds})")

    # 2. WF median Sharpe
    ms = wf.get("median_sharpe", 0.0)
    detail["wf_median_sharpe"] = ms
    if ms <= min_median_sharpe:
        fails.append(f"wf_median_sharpe<={min_median_sharpe} (got {ms})")

    # 3. WF positive folds
    pf = wf.get("pct_positive_folds", 0.0)
    detail["wf_pct_positive_folds"] = pf
    if pf < min_positive_folds:
        fails.append(f"wf_pct_positive_folds<{min_positive_folds} (got {pf})")

    # 4. Holdout return + profit factor + drawdown
    detail["ho_total_return"] = holdout_metrics.get("total_return", 0.0)
    if detail["ho_total_return"] <= 0:
        fails.append(f"holdout_total_return<=0 (got {detail['ho_total_return']})")
    pfm = holdout_metrics.get("profit_factor", 0.0)
    detail["ho_profit_factor"] = pfm
    if pfm <= min_profit_factor:
        fails.append(f"profit_factor<={min_profit_factor} (got {pfm})")
    dd = holdout_metrics.get("max_drawdown", 0.0)
    detail["ho_max_drawdown"] = dd
    if dd < max_drawdown_floor:
        fails.append(f"max_drawdown<{max_drawdown_floor} (got {dd})")
    biggest = holdout_metrics.get("biggest_trade_share", 1.0)
    detail["ho_biggest_trade_share"] = biggest
    if biggest > max_biggest_share:
        fails.append(f"biggest_trade_share>{max_biggest_share} (got {biggest})")

    # 5. Statistical significance
    detail["shuffle_p_value"] = stat_shuffle.get("p_value", 1.0)
    if not stat_shuffle.get("passes", False):
        fails.append(f"shuffle_test failed (p={stat_shuffle.get('p_value')})")
    detail["random_baseline_p_value"] = stat_random.get("p_value", 1.0)
    if not stat_random.get("passes", False):
        fails.append(f"random_baseline failed (p={stat_random.get('p_value')})")

    # 6. Execution stress
    detail["stress_total_return"] = holdout_stress.get("total_return", 0.0)
    if detail["stress_total_return"] <= 0:
        fails.append(f"stress_total_return<=0 (got {detail['stress_total_return']})")

    # 7. Yearly consistency
    detail["yearly_positive"] = yearly.get("n_positive_years", 0)
    detail["yearly_total"] = yearly.get("n_years", 0)
    if not yearly.get("passes_yearly_consistency", False):
        fails.append(f"yearly_consistency failed ({yearly.get('n_positive_years')} of {yearly.get('n_years')})")

    return CertResult(certified=not fails, failures=fails, detail=detail)
