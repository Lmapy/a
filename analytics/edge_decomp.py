"""Edge decomposition.

Given a list of Trade objects, decompose the edge into:

  - signal_quality:  hit rate (P(direction correct)) — pure signal
  - entry_quality:   how close to MFE we entered — entry timing
  - exit_quality:    realised return / MFE — exit timing

Output is one row of contribution percentages per metric. Useful for
the Alpha Judge: when entry_quality is low and signal_quality is high,
the strategy has alpha but bad fills; when signal_quality is at 50%
the strategy has no alpha at all.
"""
from __future__ import annotations

import numpy as np

from core.types import Trade


def decompose(trades: list[Trade]) -> dict:
    if not trades:
        return {}
    rets = np.array([t.ret for t in trades], dtype=float)
    mfe  = np.array([t.mfe  for t in trades], dtype=float)
    mae  = np.array([t.mae  for t in trades], dtype=float)
    entries = np.array([t.entry for t in trades], dtype=float)

    # signal_quality: how often the trade ends in profit at all
    sq = float((rets > 0).mean())

    # entry_quality: where in [MAE, MFE] swing the entry was
    rng = mae + mfe
    with np.errstate(divide="ignore", invalid="ignore"):
        entry_share = np.where(rng > 0, mae / rng, 0.5)
    eq = float(1.0 - entry_share.mean())   # closer to 1.0 = entered near worst

    # exit_quality: realised gross-of-cost return / MFE potential
    realised = rets * entries        # absolute pnl in price units
    with np.errstate(divide="ignore", invalid="ignore"):
        capture = np.where(mfe > 0, realised / mfe, np.nan)
    xq = float(np.nanmean(capture)) if np.isfinite(capture).any() else 0.0

    return {
        "signal_quality_winrate": round(sq, 4),
        "entry_quality_score":    round(eq, 4),    # 1.0 = perfect entry timing
        "exit_quality_capture":   round(xq, 4),    # 1.0 = exited at MFE
    }
