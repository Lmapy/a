"""zone_midpoint_limit — limit at the midpoint of prev-H4 body (open->close).

Different from fib 0.5 of the *range*: this uses the body, which makes
sense when most of the prior bar was a candle wick.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from entry_models.base import EntryFill
from entry_models.registry import register


@register("zone_midpoint_limit")
def fit(h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
        direction: int, params: dict) -> EntryFill | None:
    if sub is None or len(sub) == 0 or prev_h4 is None:
        return None
    o, c = float(prev_h4["open"]), float(prev_h4["close"])
    if not (math.isfinite(o) and math.isfinite(c)):
        return None
    target = (o + c) / 2.0
    if direction > 0:
        hits = np.where(sub["low"].values <= target)[0]
    else:
        hits = np.where(sub["high"].values >= target)[0]
    if len(hits) == 0:
        return None
    i = int(hits[0])
    return EntryFill(sub_idx=i, price=float(target), kind="limit",
                     notes="limit @ prev H4 body midpoint")
