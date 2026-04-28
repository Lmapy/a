"""MyFundedFutures prop firm challenge rules.

MyFundedFutures offers both intraday and EOD trailing drawdown variants
depending on the account type. The standard "Starter" plans use
TRAILING_EOD (HWM updated only at end of day using realized balance).
The "Express" plans mirror TopStep and use TRAILING_INTRADAY.

Reference: https://www.myfundedfutures.com (as of early 2026)
Verify current rules at the firm's website before live trading.
"""
from .base import AccountTier, DrawdownType, FirmRules

# Standard plans: EOD trailing drawdown
MFF_RULES = FirmRules(
    firm_name="MyFundedFutures",
    drawdown_type=DrawdownType.TRAILING_EOD,
    tiers=[
        AccountTier(
            name="25K",
            starting_balance=25_000.0,
            profit_target=1_500.0,
            max_trailing_drawdown=1_500.0,
            daily_loss_limit=1_000.0,
            min_trading_days=5,
            max_contracts=5,
        ),
        AccountTier(
            name="50K",
            starting_balance=50_000.0,
            profit_target=3_000.0,
            max_trailing_drawdown=2_500.0,
            daily_loss_limit=1_500.0,
            min_trading_days=5,
            max_contracts=10,
        ),
        AccountTier(
            name="100K",
            starting_balance=100_000.0,
            profit_target=6_000.0,
            max_trailing_drawdown=4_000.0,
            daily_loss_limit=2_500.0,
            min_trading_days=5,
            max_contracts=20,
        ),
        AccountTier(
            name="150K",
            starting_balance=150_000.0,
            profit_target=9_000.0,
            max_trailing_drawdown=5_000.0,
            daily_loss_limit=3_500.0,
            min_trading_days=5,
            max_contracts=30,
        ),
        AccountTier(
            name="200K",
            starting_balance=200_000.0,
            profit_target=12_000.0,
            max_trailing_drawdown=6_000.0,
            daily_loss_limit=4_500.0,
            min_trading_days=5,
            max_contracts=40,
        ),
    ],
)
