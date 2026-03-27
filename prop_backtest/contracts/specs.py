"""Futures contract specifications."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """Specification for a futures contract."""
    symbol: str                  # e.g. "ES"
    yfinance_ticker: str         # e.g. "ES=F"
    point_value: float           # dollars per full point (ES = $50)
    tick_size: float             # minimum price increment (ES = 0.25)
    tick_value: float            # dollars per tick = tick_size * point_value
    margin_per_contract: float   # approximate intraday margin per contract
    description: str = ""

    def pnl(self, entry_price: float, exit_price: float, contracts: int, is_short: bool = False) -> float:
        """Calculate gross PnL for a trade in dollars using tick arithmetic."""
        price_delta = exit_price - entry_price
        if is_short:
            price_delta = -price_delta
        ticks = round(price_delta / self.tick_size)
        return ticks * self.tick_value * abs(contracts)


CONTRACT_REGISTRY: dict[str, ContractSpec] = {
    "ES": ContractSpec(
        symbol="ES",
        yfinance_ticker="ES=F",
        point_value=50.0,
        tick_size=0.25,
        tick_value=12.50,
        margin_per_contract=500.0,
        description="E-mini S&P 500",
    ),
    "NQ": ContractSpec(
        symbol="NQ",
        yfinance_ticker="NQ=F",
        point_value=20.0,
        tick_size=0.25,
        tick_value=5.00,
        margin_per_contract=1000.0,
        description="E-mini Nasdaq-100",
    ),
    "CL": ContractSpec(
        symbol="CL",
        yfinance_ticker="CL=F",
        point_value=1000.0,
        tick_size=0.01,
        tick_value=10.00,
        margin_per_contract=2000.0,
        description="Crude Oil",
    ),
    "GC": ContractSpec(
        symbol="GC",
        yfinance_ticker="GC=F",
        point_value=100.0,
        tick_size=0.10,
        tick_value=10.00,
        margin_per_contract=1500.0,
        description="Gold",
    ),
    "RTY": ContractSpec(
        symbol="RTY",
        yfinance_ticker="RTY=F",
        point_value=50.0,
        tick_size=0.10,
        tick_value=5.00,
        margin_per_contract=500.0,
        description="E-mini Russell 2000",
    ),
    "MES": ContractSpec(
        symbol="MES",
        yfinance_ticker="MES=F",
        point_value=5.0,
        tick_size=0.25,
        tick_value=1.25,
        margin_per_contract=50.0,
        description="Micro E-mini S&P 500",
    ),
    "MNQ": ContractSpec(
        symbol="MNQ",
        yfinance_ticker="MNQ=F",
        point_value=2.0,
        tick_size=0.25,
        tick_value=0.50,
        margin_per_contract=100.0,
        description="Micro E-mini Nasdaq-100",
    ),
}


def get_contract(symbol: str) -> ContractSpec:
    """Look up a contract by symbol (case-insensitive)."""
    key = symbol.upper()
    if key not in CONTRACT_REGISTRY:
        available = ", ".join(CONTRACT_REGISTRY.keys())
        raise ValueError(f"Unknown contract '{symbol}'. Available: {available}")
    return CONTRACT_REGISTRY[key]
