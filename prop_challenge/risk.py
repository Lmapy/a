"""Risk-sizing models.

A RiskModel converts a Trade.pnl in price units into a dollar PnL given
a contract spec and current account state.

  fixed_micros        always trade `n` contracts (each = $/point)
  fixed_dollar_risk   target a fixed $ risk per trade by sizing contracts
                      to the trade's MAE distance (post-hoc, since we
                      don't know the future stop distance) -- proxied as
                      a fixed dollar bet per trade
  pct_dd_buffer       size to risk X% of remaining drawdown buffer
  vol_adjusted        scale inversely with recent ATR
  reduce_after_loss   half-size after the most recent trade was a loss
  scale_after_high    +1 size after each new equity high
"""
from __future__ import annotations

from dataclasses import dataclass

from prop_challenge.accounts import AccountSpec, dollar_per_price_unit


@dataclass
class RiskModel:
    name: str
    instrument: str = "MGC"
    contracts_base: int = 1
    dollar_risk_per_trade: float | None = None
    pct_dd_buffer: float | None = None
    vol_adjust: bool = False
    reduce_after_loss: bool = False
    scale_after_high: bool = False
    contracts_max: int = 5

    def size(self, *, balance: float, starting_balance: float,
             max_loss: float, last_trade_loss: bool, equity_high: bool,
             trade_pnl_price: float) -> int:
        """Return integer contract count to size this trade."""
        c = self.contracts_base
        if self.pct_dd_buffer is not None:
            buffer_left = balance - (starting_balance - max_loss)
            target_risk = max(0.0, buffer_left * self.pct_dd_buffer / 100.0)
            dpp = dollar_per_price_unit(self.instrument)
            # one contract = ~|trade_pnl_price * dpp|
            per_contract_risk = max(1.0, abs(trade_pnl_price) * dpp)
            c = max(1, int(target_risk / per_contract_risk))
        if self.dollar_risk_per_trade is not None:
            dpp = dollar_per_price_unit(self.instrument)
            per_contract_risk = max(1.0, abs(trade_pnl_price) * dpp)
            c = max(1, int(self.dollar_risk_per_trade / per_contract_risk))
        if self.reduce_after_loss and last_trade_loss:
            c = max(1, c // 2)
        if self.scale_after_high and equity_high:
            c += 1
        return max(1, min(self.contracts_max, c))


def all_models(account: AccountSpec) -> list[RiskModel]:
    cap = max(1, min(account.max_contracts, 5))
    return [
        RiskModel(name="micro_1",                contracts_base=1,
                  contracts_max=cap),
        RiskModel(name="micro_2",                contracts_base=2,
                  contracts_max=cap),
        RiskModel(name="dollar_risk_50",         dollar_risk_per_trade=50.0,
                  contracts_max=cap),
        RiskModel(name="dollar_risk_100",        dollar_risk_per_trade=100.0,
                  contracts_max=cap),
        RiskModel(name="pct_dd_buffer_2pct",     pct_dd_buffer=2.0,
                  contracts_max=cap),
        RiskModel(name="reduce_after_loss",      contracts_base=2,
                  reduce_after_loss=True, contracts_max=cap),
        RiskModel(name="scale_after_high",       contracts_base=1,
                  scale_after_high=True, contracts_max=cap),
    ]
