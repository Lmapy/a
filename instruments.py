"""Multi-instrument configuration for prop firm strategy.

Each instrument has different tick values, margin, and contract specs.
The strategy adapts risk per trade based on instrument characteristics.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentConfig:
    name: str           # display name
    symbol: str         # short symbol
    tick_size: float    # minimum price increment
    tick_value: float   # dollar value per tick per micro contract
    oz_per_contract: int  # multiplier (oz for gold, 1 for index points)
    data_file: str      # parquet file name
    # Trading characteristics
    avg_atr_5min: float  # approximate 5-min ATR for position sizing
    session_hours: tuple  # best trading hours (UTC)


# Micro Gold Futures (MGC)
GOLD = InstrumentConfig(
    name="Micro Gold", symbol="MGC",
    tick_size=0.10, tick_value=1.0, oz_per_contract=10,
    data_file="XAUUSD_1min.parquet",
    avg_atr_5min=0.90,
    session_hours=(0, 12),  # Asia + London + early NY
)

# Micro E-mini Nasdaq (MNQ)
NQ = InstrumentConfig(
    name="Micro NQ", symbol="MNQ",
    tick_size=0.25, tick_value=0.50, oz_per_contract=1,
    data_file="NQ_1min.parquet",
    avg_atr_5min=15.0,
    session_hours=(13, 20),  # NY session (13:00-20:00 UTC = 8AM-3PM ET)
)

# Micro E-mini S&P 500 (MES)
ES = InstrumentConfig(
    name="Micro ES", symbol="MES",
    tick_size=0.25, tick_value=1.25, oz_per_contract=1,
    data_file="ES_1min.parquet",
    avg_atr_5min=5.0,
    session_hours=(13, 20),  # NY session
)

# Micro WTI Crude Oil (MCL)
CL = InstrumentConfig(
    name="Micro CL", symbol="MCL",
    tick_size=0.01, tick_value=1.0, oz_per_contract=100,
    data_file="CL_1min.parquet",
    avg_atr_5min=0.15,
    session_hours=(13, 19),  # NY session (crude most active 8AM-2PM ET)
)

ALL_INSTRUMENTS = [GOLD, NQ, ES, CL]
