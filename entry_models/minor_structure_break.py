"""minor_structure_break — enter on the first M15 bar that breaks the high
of the prior n M15 bars (long; symmetric for short).

params:
  lookback (int)  number of M15 bars to take the structural extreme from. Default 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from entry_models.base import EntryFill
from entry_models.registry import register


@register("minor_structure_break")
def fit(h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
        direction: int, params: dict) -> EntryFill | None:
    if sub is None or len(sub) == 0:
        return None
    lookback = int(params.get("lookback", 3))
    highs = sub["high"].values
    lows = sub["low"].values
    closes = sub["close"].values
    for i in range(lookback, len(sub)):
        ref_hi = highs[i - lookback:i].max()
        ref_lo = lows[i - lookback:i].min()
        if direction > 0 and closes[i] > ref_hi:
            return EntryFill(sub_idx=int(i), price=float(closes[i]), kind="market",
                             notes=f"break high of prev {lookback} bars")
        if direction < 0 and closes[i] < ref_lo:
            return EntryFill(sub_idx=int(i), price=float(closes[i]), kind="market",
                             notes=f"break low of prev {lookback} bars")
    return None
