"""Position sizing for prop firm compliance."""

from config import CONTRACT, PARAMS, RULES


def calculate_position_size(
    sl_distance_dollars: float,
    risk_per_trade: float = PARAMS.risk_per_trade,
) -> int:
    """Calculate the number of MGC contracts to trade.

    contracts = risk_amount / (sl_distance_in_ticks * tick_value)

    Args:
        sl_distance_dollars: Distance from entry to stop loss in dollars.
        risk_per_trade: Maximum dollar risk per trade.

    Returns:
        Number of contracts (capped at prop firm max).
    """
    if sl_distance_dollars <= 0:
        return 0

    sl_ticks = sl_distance_dollars / CONTRACT.tick_size
    dollar_risk_per_contract = sl_ticks * CONTRACT.tick_value

    if dollar_risk_per_contract <= 0:
        return 0

    contracts = int(risk_per_trade / dollar_risk_per_contract)
    contracts = max(1, contracts)
    contracts = min(contracts, RULES.max_micros)

    return contracts


def calculate_trade_pnl(
    entry_price: float,
    exit_price: float,
    contracts: int,
    is_long: bool,
    slippage_ticks: int = PARAMS.slippage_ticks,
) -> float:
    """Calculate P&L for a trade including slippage and commissions.

    Returns net P&L in dollars.
    """
    slippage = slippage_ticks * CONTRACT.tick_size
    commission = PARAMS.commission_per_side * 2 * contracts  # round trip

    if is_long:
        # Slippage: worse entry (higher), worse exit (lower)
        effective_entry = entry_price + slippage
        effective_exit = exit_price - slippage
        raw_pnl = (effective_exit - effective_entry) * CONTRACT.oz_per_contract * contracts
    else:
        effective_entry = entry_price - slippage
        effective_exit = exit_price + slippage
        raw_pnl = (effective_entry - effective_exit) * CONTRACT.oz_per_contract * contracts

    return raw_pnl - commission
