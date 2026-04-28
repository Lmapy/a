"""delayed_entry_N — enter at the open of the (N)th M15 sub-bar.

Two registrations:
  delayed_entry_1   N=1
  delayed_entry_2   N=2

params (optional, overrides N):
  n (int)
"""
from __future__ import annotations

import pandas as pd

from entry_models.base import EntryFill
from entry_models.registry import register


def _delayed(n_default: int):
    def fit(h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
            direction: int, params: dict) -> EntryFill | None:
        if sub is None or len(sub) == 0:
            return None
        n = int(params.get("n", n_default))
        if n >= len(sub):
            return None
        return EntryFill(sub_idx=int(n), price=float(sub["open"].iloc[n]),
                         kind="market", notes=f"open of sub-bar #{n}")
    return fit


register("delayed_entry_1")(_delayed(1))
register("delayed_entry_2")(_delayed(2))
