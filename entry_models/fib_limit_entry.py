"""fib_limit_entry — limit order at a fib retracement of the previous H4 candle.

params:
  level (float)  fib level in [0, 1]; default 0.5
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from entry_models.base import EntryFill
from entry_models.registry import register


@register("fib_limit_entry")
def fit(h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
        direction: int, params: dict) -> EntryFill | None:
    if sub is None or len(sub) == 0 or prev_h4 is None:
        return None
    ph, pl = float(prev_h4["high"]), float(prev_h4["low"])
    if not (math.isfinite(ph) and math.isfinite(pl)) or ph <= pl:
        return None
    rng = ph - pl
    level = float(params.get("level", 0.5))
    target = ph - level * rng if direction > 0 else pl + level * rng
    if direction > 0:
        hits = np.where(sub["low"].values <= target)[0]
    else:
        hits = np.where(sub["high"].values >= target)[0]
    if len(hits) == 0:
        return None
    i = int(hits[0])
    return EntryFill(sub_idx=i, price=float(target), kind="limit",
                     notes=f"limit @ fib {level} of prev H4 range")
