"""touch_entry — enter at the open of the first sub-bar of the new H4."""
from __future__ import annotations

import pandas as pd

from entry_models.base import EntryFill
from entry_models.registry import register


@register("touch_entry")
def fit(h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
        direction: int, params: dict) -> EntryFill | None:
    if sub is None or len(sub) == 0:
        return None
    return EntryFill(sub_idx=0, price=float(sub["open"].iloc[0]), kind="market",
                     notes="market open of first sub-bar")
