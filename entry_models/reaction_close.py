"""reaction_close — wait for the first sub-bar that closes in trade direction; enter at its close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from entry_models.base import EntryFill
from entry_models.registry import register


@register("reaction_close")
def fit(h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
        direction: int, params: dict) -> EntryFill | None:
    if sub is None or len(sub) == 0:
        return None
    color = np.sign(sub["close"].values - sub["open"].values).astype(int)
    hits = np.where(color == direction)[0]
    if len(hits) == 0:
        return None
    i = int(hits[0])
    return EntryFill(sub_idx=i, price=float(sub["close"].iloc[i]), kind="market",
                     notes=f"first confirming sub-bar close (idx={i})")
