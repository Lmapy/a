"""Position sizing for prop firm compliance with adaptive risk scaling."""

from config import CONTRACT, PARAMS, RULES


def calculate_position_size(
    sl_distance_dollars: float,
    risk_per_trade: float = PARAMS.risk_per_trade,
) -> int:
    """Calculate the number of MGC contracts to trade."""
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
    base_risk: float = PARAMS.risk_per_trade,
) -> float:
    """Adaptively scale risk based on progress toward profit target.

    Strategy:
    - If ahead of pace: maintain or slightly reduce risk (protect gains)
    - If behind pace: gradually increase risk (catch up)
    - Never exceed $500 per trade (hard cap for safety)
    - In first 5 days: use base risk (build cushion)
    """
    if trading_days < 1:
        return base_risk

    days_remaining = max(1, max_trading_days - trading_days)
    target = RULES.profit_target
    pnl_needed = target - cumulative_pnl
    daily_rate_needed = pnl_needed / days_remaining

    # Expected daily profit at base risk (based on strategy stats: ~$22/day per $250 risk)
    # PF 1.54, 1.4 trades/day, 47% WR, 2.5:1 R:R
    expected_daily_at_base = base_risk * 0.09  # ~9% of risk per day expected

    # First 5 days: use base risk, establish a track record
    if trading_days <= 5:
        return base_risk

    # Calculate risk multiplier
    if cumulative_pnl >= target:
        return base_risk  # already passed

    if pnl_needed <= 0:
        return base_risk

    # How aggressive do we need to be?
    pace_ratio = daily_rate_needed / max(expected_daily_at_base, 1)

    if pace_ratio <= 1.0:
        # On pace or ahead - maintain base risk
        risk = base_risk
    elif pace_ratio <= 2.0:
        # Slightly behind - increase by 10-25%
        risk = base_risk * (1.0 + (pace_ratio - 1.0) * 0.25)
    elif pace_ratio <= 4.0:
        # Behind - increase by 25-40%
        risk = base_risk * 1.25 + (pace_ratio - 2.0) * base_risk * 0.075
    else:
        # Far behind - moderate aggression, capped at 1.5x
        risk = base_risk * 1.5

    # Safety caps
    risk = min(risk, base_risk * 1.5)  # never exceed 1.5x base risk
    risk = max(risk, base_risk * 0.75)  # never go below 75% base risk

    # If in drawdown (negative P&L), DON'T increase risk - reduce it
    if cumulative_pnl < -500:
        risk = base_risk * 0.8  # reduce when deep in the hole

    # If we have profit cushion, allow slightly more
    if cumulative_pnl > 1500:
        risk = min(risk * 1.15, base_risk * 1.5)

    return risk


def calculate_trade_pnl(
    entry_price: float,
    exit_price: float,
    contracts: int,
    is_long: bool,
    slippage_ticks: int = PARAMS.slippage_ticks,
) -> float:
    """Calculate P&L for a trade including slippage and commissions."""
    slippage = slippage_ticks * CONTRACT.tick_size
    commission = PARAMS.commission_per_side * 2 * contracts

    if is_long:
        effective_entry = entry_price + slippage
        effective_exit = exit_price - slippage
        raw_pnl = (effective_exit - effective_entry) * CONTRACT.oz_per_contract * contracts
    else:
        effective_entry = entry_price - slippage
        effective_exit = exit_price + slippage
        raw_pnl = (effective_entry - effective_exit) * CONTRACT.oz_per_contract * contracts

    return raw_pnl - commission
