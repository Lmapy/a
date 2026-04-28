"""sweep_reclaim — enter after a liquidity sweep of the prior bar's extreme followed by a reclaim.

Long after green prev: wait for an M15 bar to print a low BELOW prev_low,
                       then a subsequent M15 close back ABOVE prev_low.
Short after red:       symmetric (sweep above prev_high, close back below).

This is a "stop-hunt then continuation" entry; the entry price is the
close of the reclaiming bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from entry_models.base import EntryFill
from entry_models.registry import register


@register("sweep_reclaim")
def fit(h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
        direction: int, params: dict) -> EntryFill | None:
    if sub is None or len(sub) < 2 or prev_h4 is None:
        return None
    ph, pl = float(prev_h4["high"]), float(prev_h4["low"])
    lows = sub["low"].values
    highs = sub["high"].values
    closes = sub["close"].values
    swept_idx = -1
    for i in range(len(sub)):
        if direction > 0 and lows[i] < pl:
            swept_idx = i
            break
        if direction < 0 and highs[i] > ph:
            swept_idx = i
            break
    if swept_idx < 0:
        return None
    # reclaim: any subsequent bar whose close is back inside the prev range
    for j in range(swept_idx + 1, len(sub)):
        if direction > 0 and closes[j] > pl:
            return EntryFill(sub_idx=int(j), price=float(closes[j]), kind="market",
                             notes=f"sweep@{swept_idx} reclaim@{j}")
        if direction < 0 and closes[j] < ph:
            return EntryFill(sub_idx=int(j), price=float(closes[j]), kind="market",
                             notes=f"sweep@{swept_idx} reclaim@{j}")
    return None
