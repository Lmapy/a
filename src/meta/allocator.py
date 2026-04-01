from __future__ import annotations

import math


class RiskAllocator:
    """Distributes the total allowed contracts across active strategies.

    Given a total_contracts budget and strategy allocation percentages,
    this ensures:
        - Sum of all positions <= total_contracts
        - No strategy exceeds its allocation
        - At least 1 contract per active strategy (if total allows)
    """

    def allocate(self, total_contracts: int,
                 strategy_allocations: dict[str, float]) -> dict[str, int]:
        """Returns dict of strategy_name -> max_contracts."""
        if total_contracts <= 0 or not strategy_allocations:
            return {name: 0 for name in strategy_allocations}

        # Normalize allocations
        total_alloc = sum(strategy_allocations.values())
        if total_alloc <= 0:
            return {name: 0 for name in strategy_allocations}

        result: dict[str, int] = {}
        allocated = 0

        # First pass: floor allocation
        for name, pct in strategy_allocations.items():
            normalized = pct / total_alloc
            contracts = math.floor(total_contracts * normalized)
            contracts = max(contracts, 0)
            result[name] = contracts
            allocated += contracts

        # Distribute remaining contracts to strategies with highest fractional part
        remaining = total_contracts - allocated
        if remaining > 0:
            fractional = {}
            for name, pct in strategy_allocations.items():
                normalized = pct / total_alloc
                frac = (total_contracts * normalized) - math.floor(total_contracts * normalized)
                fractional[name] = frac

            for name in sorted(fractional, key=fractional.get, reverse=True):
                if remaining <= 0:
                    break
                result[name] += 1
                remaining -= 1

        # Ensure at least 1 contract for each active strategy if possible
        for name in result:
            if result[name] == 0 and total_contracts > sum(result.values()):
                result[name] = 1

        return result
