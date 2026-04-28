"""Risk-sizing models (Phase 7 hardening — no future leakage).

`RiskModel.size()` consumes ONLY pre-trade information:

  balance, starting_balance, max_loss      — account state at entry time
  last_trade_loss, equity_high             — known from the prior closed trade
  stop_distance_price                      — |entry - stop|, in price units;
                                              determined by the spec at fill
  atr_pre_entry                            — ATR computed on bars STRICTLY
                                              before the entry bar (fallback
                                              risk proxy when no stop)

The previous implementation accepted `trade_pnl_price` (the realised PnL of
the trade about to be sized) and used `abs(trade_pnl_price * dpp)` as the
per-contract risk. That meant the simulator sized each trade based on its
own future outcome, which inflated pass probability across the leaderboard
and made the prop-challenge results untrustworthy.

If a spec produces trades with no stop and no ATR, `dollar_risk_per_trade`
and `pct_dd_buffer` cannot size from risk and fall back to `contracts_base`.
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
    pct_dd_buffer: float | None = None      # % of remaining DD buffer to risk
    vol_adjust: bool = False                # not used in size(); kept for spec parity
    reduce_after_loss: bool = False
    scale_after_high: bool = False
    contracts_max: int = 5

    def size(self, *,
             balance: float,
             starting_balance: float,
             max_loss: float,
             last_trade_loss: bool,
             equity_high: bool,
             stop_distance_price: float | None = None,
             atr_pre_entry: float | None = None) -> int:
        """Return integer contract count to size this trade.

        Pre-trade inputs only. The realised PnL of THIS trade is
        intentionally absent from the signature -- a previous version
        accepted it and used it for sizing, which leaked future
        information into position size.
        """
        dpp = dollar_per_price_unit(self.instrument)

        # Determine per-contract risk from the explicit stop, then fall
        # back to ATR if the spec has no stop. If neither is available,
        # size models that require risk must fall back to base contracts.
        per_contract_risk: float | None = None
        if stop_distance_price is not None and stop_distance_price > 0:
            per_contract_risk = float(stop_distance_price) * dpp
        elif atr_pre_entry is not None and atr_pre_entry > 0:
            per_contract_risk = float(atr_pre_entry) * dpp

        c = self.contracts_base

        if self.dollar_risk_per_trade is not None:
            if per_contract_risk and per_contract_risk > 0:
                c = max(1, int(self.dollar_risk_per_trade / max(1.0, per_contract_risk)))
            # else: no risk denominator -> stay at contracts_base

        if self.pct_dd_buffer is not None:
            buffer_left = max(0.0, balance - (starting_balance - max_loss))
            target_risk = buffer_left * self.pct_dd_buffer / 100.0
            if per_contract_risk and per_contract_risk > 0:
                c = max(1, int(target_risk / max(1.0, per_contract_risk)))

        if self.reduce_after_loss and last_trade_loss:
            c = max(1, c // 2)
        if self.scale_after_high and equity_high:
            c += 1

        return max(1, min(self.contracts_max, c))


def all_models(account: AccountSpec) -> list[RiskModel]:
    cap = max(1, min(account.max_contracts, 5))
    return [
        RiskModel(name="micro_1", contracts_base=1, contracts_max=cap),
        RiskModel(name="micro_2", contracts_base=2, contracts_max=cap),
        RiskModel(name="dollar_risk_50",  dollar_risk_per_trade=50.0,
                  contracts_max=cap),
        RiskModel(name="dollar_risk_100", dollar_risk_per_trade=100.0,
                  contracts_max=cap),
        RiskModel(name="pct_dd_buffer_2pct", pct_dd_buffer=2.0,
                  contracts_max=cap),
        RiskModel(name="reduce_after_loss", contracts_base=2,
                  reduce_after_loss=True, contracts_max=cap),
        RiskModel(name="scale_after_high",  contracts_base=1,
                  scale_after_high=True,  contracts_max=cap),
    ]
