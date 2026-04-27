"""Constants used across v2."""
from __future__ import annotations

# H4 trading sessions in UTC. ~6 H4 bars per gold trading day, 5 days/week, 52 weeks.
H4_BARS_PER_YEAR = 1560

# Sub-bar counts inside an H4 candle (used by validators).
TF_MINUTES = {"M1": 1, "M3": 3, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}

# XAUUSD broker spec — point size for the cent symbol used in the dataset.
POINT_SIZE = 0.001

# Standard fib retracement levels.
FIB_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

# UTC hours that bound the three traditional FX sessions.
SESSION_HOURS = {
    "asia":   {0, 1, 2, 3, 4, 5, 6, 7},
    "london": {7, 8, 9, 10, 11, 12, 13, 14, 15},
    "ny":     {12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
}

# Prop firm presets used by prop.simulator.
PROP_ACCOUNTS = {
    "25k":  {"balance": 25_000,  "daily_loss_limit": 500,   "trailing_dd": 1_500,  "max_contracts": 1},
    "50k":  {"balance": 50_000,  "daily_loss_limit": 1_000, "trailing_dd": 2_500,  "max_contracts": 2},
    "150k": {"balance": 150_000, "daily_loss_limit": 3_000, "trailing_dd": 4_500,  "max_contracts": 5},
}
