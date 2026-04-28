"""Lucid Trading prop firm challenge rules.

Lucid uses TRAILING_INTRADAY drawdown similar to TopStep.
The lock-to-breakeven rule applies: once your profit exceeds the
max_trailing_drawdown, the floor locks at starting_balance.

Reference: https://www.lucidtrading.com (as of early 2026)
Verify current rules at the firm's website before live trading.
"""
from .base import AccountTier, DrawdownType, FirmRules

LUCID_RULES = FirmRules(
    firm_name="Lucid",
    drawdown_type=DrawdownType.TRAILING_INTRADAY,
    tiers=[
        AccountTier(
            name="25K",
            starting_balance=25_000.0,
            profit_target=1_500.0,
            max_trailing_drawdown=1_500.0,
            daily_loss_limit=750.0,
            min_trading_days=5,
            max_contracts=5,
        ),
        AccountTier(
            name="50K",
            starting_balance=50_000.0,
            profit_target=3_000.0,
            max_trailing_drawdown=2_500.0,
            daily_loss_limit=1_250.0,
            min_trading_days=5,
            max_contracts=10,
        ),
        AccountTier(
            name="100K",
            starting_balance=100_000.0,
            profit_target=6_000.0,
            max_trailing_drawdown=4_000.0,
            daily_loss_limit=2_000.0,
            min_trading_days=5,
            max_contracts=20,
        ),
        AccountTier(
            name="150K",
            starting_balance=150_000.0,
            profit_target=8_000.0,
            max_trailing_drawdown=5_000.0,
            daily_loss_limit=2_500.0,
            min_trading_days=5,
            max_contracts=30,
        ),
    ],
)
