"""Base contract for entry models.

An entry model receives:
  - the H4 bar that just closed (signal context, includes OHLC and prev bar)
  - the slice of lower-TF bars (M15 etc.) that belong to the *next* H4 bucket
  - the trade direction (already resolved by the signal)

It returns a single fill candidate (entry_idx, entry_price, fill_kind)
or None if no fill should be taken. It does NOT model slippage, spread,
or stops -- those live in the executor.

This separation lets the same M15 stream be re-played through every
entry model without duplicating execution code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class EntryFill:
    sub_idx: int                 # index inside the H4 bucket (0 = first M15)
    price: float
    kind: str                    # "market" | "limit" | "stop"
    notes: str = ""


class EntryModel(Protocol):
    name: str

    def fit(self, h4_row: pd.Series, prev_h4: pd.Series, sub: pd.DataFrame,
            direction: int, params: dict) -> EntryFill | None: ...
