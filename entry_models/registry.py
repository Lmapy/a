"""Registry mapping entry-model name -> fit function.

Importing this module also imports each model file so its decorator
registers it. To add a new entry model: drop a file in entry_models/
and call register("name") on its fit function.
"""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from entry_models.base import EntryFill

_REGISTRY: Dict[str, Callable] = {}


def register(name: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"entry model already registered: {name}")
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown entry model: {name} (have: {list(_REGISTRY)})")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY.keys())


def fit(name: str, h4_row: pd.Series, prev_h4: pd.Series,
        sub: pd.DataFrame, direction: int, params: dict) -> EntryFill | None:
    return get(name)(h4_row, prev_h4, sub, direction, params)


# Force-register all built-in models on import.
from entry_models import (  # noqa: E402, F401
    touch_entry,
    fib_limit_entry,
    reaction_close,
    zone_midpoint_limit,
    sweep_reclaim,
    minor_structure_break,
    delayed_entry,
)
