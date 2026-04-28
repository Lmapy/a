"""TopStep (TopstepTrader) prop firm challenge rules.

TopStep uses TRAILING_INTRADAY drawdown:
- The drawdown floor = peak_intraday_equity - max_trailing_drawdown
- Once the account is profitable enough that the raw floor would exceed
  starting_balance, the floor is CAPPED at starting_balance (lock-to-breakeven).
  This means the trailing drawdown stops trailing once you have built a cushion
  equal to max_trailing_drawdown above the starting balance.

Reference: https://www.topstep.com/funded-accounts/ (as of early 2026)
"""
from .base import AccountTier, DrawdownType, FirmRules

TOPSTEP_RULES = FirmRules(
    firm_name="TopStep",
    drawdown_type=DrawdownType.TRAILING_INTRADAY,
    tiers=[
        AccountTier(
            name="50K",
            starting_balance=50_000.0,
            profit_target=3_000.0,
            max_trailing_drawdown=2_000.0,
            daily_loss_limit=1_000.0,
            min_trading_days=10,
            max_contracts=10,
        ),
        AccountTier(
            name="100K",
            starting_balance=100_000.0,
            profit_target=6_000.0,
            max_trailing_drawdown=3_000.0,
            daily_loss_limit=2_000.0,
            min_trading_days=10,
            max_contracts=20,
        ),
        AccountTier(
            name="150K",
            starting_balance=150_000.0,
            profit_target=9_000.0,
            max_trailing_drawdown=4_500.0,
            daily_loss_limit=3_000.0,
            min_trading_days=10,
            max_contracts=30,
        ),
    ],
)
