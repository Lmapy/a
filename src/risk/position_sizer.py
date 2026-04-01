from __future__ import annotations

import logging
import math

from src.core.config import FirmConfig, RiskConfig
from src.data.models import INSTRUMENT_SPECS

logger = logging.getLogger(__name__)


class PositionSizer:
    """Dynamic position sizing based on remaining drawdown budget.

    Core formula:
        remaining_budget = current_balance - drawdown_floor
        risk_per_trade = remaining_budget * risk_pct
        contracts = floor(risk_per_trade / (stop_ticks * tick_value))

    Scaling modes:
        - Normal: standard risk_pct
        - Protection: risk_pct * 0.5 when budget < scale_down_threshold
        - Acceleration: risk_pct * 1.3 when profit > 50% of target (finish faster)
        - Lockout: 0 contracts when budget < min_drawdown_buffer
    """

    def __init__(self, firm_config: FirmConfig, risk_config: RiskConfig):
        self.firm = firm_config
        self.risk = risk_config

    def calculate(
        self,
        instrument: str,
        stop_distance_ticks: float,
        remaining_budget: float,
        profit_so_far: float,
    ) -> int:
        """Calculate number of contracts for a trade.

        Args:
            instrument: Futures symbol (ES, NQ, MES, MNQ)
            stop_distance_ticks: Number of ticks to stop loss
            remaining_budget: current_balance - drawdown_floor
            profit_so_far: cumulative profit toward target

        Returns:
            Number of contracts (0 = don't trade)
        """
        # Hard lockout: too close to drawdown floor
        if remaining_budget <= self.risk.min_drawdown_buffer:
            logger.warning(
                f"LOCKOUT: remaining_budget ${remaining_budget:.0f} "
                f"< min buffer ${self.risk.min_drawdown_buffer:.0f}"
            )
            return 0

        if stop_distance_ticks <= 0:
            return 0

        spec = INSTRUMENT_SPECS.get(instrument)
        if not spec:
            logger.error(f"Unknown instrument: {instrument}")
            return 0

        tick_value = spec["tick_value"]

        # Determine effective risk percentage
        risk_pct = self.risk.max_risk_per_trade_pct / 100.0

        # Protection mode: reduce size when budget is thin
        if remaining_budget < self.risk.scale_down_threshold:
            risk_pct *= 0.5
            logger.debug(f"Protection mode: risk_pct halved to {risk_pct:.4f}")

        # Acceleration mode: increase size when well ahead
        elif profit_so_far >= self.firm.profit_target * (self.risk.acceleration_profit_pct / 100.0):
            risk_pct *= self.risk.acceleration_multiplier
            logger.debug(f"Acceleration mode: risk_pct x{self.risk.acceleration_multiplier}")

        # Calculate risk dollars and contracts
        risk_dollars = remaining_budget * risk_pct
        stop_dollars = stop_distance_ticks * tick_value

        contracts = math.floor(risk_dollars / stop_dollars)

        # Always allow at least 1 contract if budget is above min buffer
        # and the stop risk is within 50% of remaining budget
        if contracts == 0 and remaining_budget > self.risk.min_drawdown_buffer:
            if stop_dollars <= remaining_budget * 0.50:
                contracts = 1

        # Clamp to firm maximum
        max_contracts = self.firm.max_contracts.get(instrument, 1)
        contracts = min(contracts, max_contracts)

        # Never go below 0
        contracts = max(contracts, 0)

        # Safety: never risk more than 30% of remaining budget in one trade
        max_risk = remaining_budget * 0.30
        actual_risk = contracts * stop_dollars
        if actual_risk > max_risk and contracts > 1:
            contracts = math.floor(max_risk / stop_dollars)
            contracts = max(contracts, 0)

        logger.debug(
            f"Size calc: {instrument} stop={stop_distance_ticks}t, "
            f"budget=${remaining_budget:.0f}, risk_pct={risk_pct:.4f}, "
            f"contracts={contracts}"
        )
        return contracts
