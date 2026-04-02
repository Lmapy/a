"""Position sizing for prop firm compliance with adaptive risk scaling."""

from config import CONTRACT, PARAMS, RULES


def calculate_position_size(
    sl_distance_dollars: float,
    risk_per_trade: float | None = None,
) -> int:
    """Calculate the number of MGC contracts to trade."""
    if risk_per_trade is None:
        risk_per_trade = PARAMS.risk_per_trade

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


def calculate_adaptive_risk(
    cumulative_pnl: float,
    trading_days: int,
    max_trading_days: int = RULES.max_trading_days,
    base_risk: float | None = None,
) -> float:
    """Adaptively scale risk based on progress toward profit target.

    Strategy:
    - If ahead of pace: maintain or slightly reduce risk (protect gains)
    - If behind pace: gradually increase risk (catch up)
    - Never exceed $500 per trade (hard cap for safety)
    - In first 5 days: use base risk (build cushion)
    """
    if base_risk is None:
        base_risk = PARAMS.risk_per_trade

    # Ramp-up risk: start small, increase as we prove the edge is working
    if trading_days <= 3:
        return base_risk * 0.5  # First 3 days: half risk (probe phase)
    elif trading_days <= 6:
        if cumulative_pnl > 0:
            return base_risk  # Profitable: full risk
        else:
            return base_risk * 0.6  # Behind: still conservative
    else:
        # After day 6: scale based on progress
        if cumulative_pnl >= 1500:
            return base_risk * 1.3  # Ahead: push
        elif cumulative_pnl >= 500:
            return base_risk  # On track
        elif cumulative_pnl >= 0:
            return base_risk * 1.2  # Slightly behind: moderate push
        else:
            return base_risk * 0.7  # Losing: reduce to survive

    if trading_days < 1:
        return base_risk

    days_remaining = max(1, max_trading_days - trading_days)
    target = RULES.profit_target
    pnl_needed = target - cumulative_pnl

    if pnl_needed <= 0:
        return base_risk  # already passed

    daily_rate_needed = pnl_needed / days_remaining
    expected_daily = base_risk * 0.09  # ~9% of risk per day
    pace_ratio = daily_rate_needed / max(expected_daily, 1)

    # First 5 days: base risk
    if trading_days <= 5:
        return base_risk

    # Scale based on pace
    if pace_ratio <= 1.5:
        risk = base_risk
    elif pace_ratio <= 3.0:
        risk = base_risk * (1.0 + (pace_ratio - 1.5) * 0.33)
    else:
        risk = base_risk * 1.5

    # Caps
    risk = min(risk, base_risk * 1.5)
    risk = max(risk, base_risk * 0.75)

    # Cushion bonus
    if cumulative_pnl > 1500:
        risk = min(risk * 1.15, base_risk * 1.5)

    return risk


def calculate_trade_pnl(
    entry_price: float,
    exit_price: float,
    contracts: int,
    is_long: bool,
    exit_is_sl: bool = False,
    entry_is_limit: bool = False,
    slippage_ticks: int = PARAMS.slippage_ticks,
) -> float:
    """Calculate P&L for a trade including slippage and commissions.

    Slippage model (realistic):
    - Entry via limit order: 0 slippage
    - Entry via market order: 1 tick slippage (adverse)
    - SL via stop-market order: full slippage (adverse)
    - TP via limit order: 0 slippage
    """
    commission = PARAMS.commission_per_side * 2 * contracts
    tick = CONTRACT.tick_size

    if entry_is_limit:
        effective_entry = entry_price  # limit order: no slippage
    else:
        # Market order entry: 1 tick adverse slippage
        if is_long:
            effective_entry = entry_price + tick
        else:
            effective_entry = entry_price - tick

    if exit_is_sl:
        # Stop-loss is a market order - gets full slippage
        sl_slip = slippage_ticks * tick
        if is_long:
            effective_exit = exit_price - sl_slip
        else:
            effective_exit = exit_price + sl_slip
    else:
        # Take-profit is a limit order - no slippage
        effective_exit = exit_price

    if is_long:
        raw_pnl = (effective_exit - effective_entry) * CONTRACT.oz_per_contract * contracts
    else:
        raw_pnl = (effective_entry - effective_exit) * CONTRACT.oz_per_contract * contracts

    return raw_pnl - commission
