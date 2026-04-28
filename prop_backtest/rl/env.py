"""PropFirmEnv — Gymnasium environment wrapping the backtest engine."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "gymnasium is required. Install with: pip install gymnasium"
    ) from e

from prop_backtest.account.monitor import RuleMonitor
from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.engine.broker import Signal, SimulatedBroker
from prop_backtest.firms.base import FirmRules
from prop_backtest.rl.features import build_observation, observation_size
from prop_backtest.rl.rewards import RewardShaper


class PropFirmEnv(gym.Env):
    """A Gymnasium environment that simulates a prop firm challenge.

    Action space: Discrete(4)
        0 = hold
        1 = buy 1 contract (or close short)
        2 = sell 1 contract (or close long)
        3 = close position

    Observation space: Box(obs_dim,)
        See prop_backtest.rl.features.build_observation for layout.

    Episode terminates when:
        - A prop firm rule is violated (trailing DD, daily loss).
        - The profit target is hit.
        - All bars are exhausted (truncated=True).

    Args:
        bars: Historical bar data (will be replayed each episode).
        firm_rules: FirmRules instance.
        tier_name: Account tier, e.g. "100K".
        contract: ContractSpec.
        commission_per_rt: Round-turn commission per contract.
        slippage_ticks: Number of adverse ticks of slippage on fills.
        reward_shaper: Optional custom RewardShaper.
        window: Number of bars in the observation window.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: list[BarData],
        firm_rules: FirmRules,
        tier_name: str,
        contract: ContractSpec,
        commission_per_rt: float = 4.50,
        slippage_ticks: int = 0,
        reward_shaper: Optional[RewardShaper] = None,
        window: int = 20,
    ) -> None:
        super().__init__()

        self.bars = bars
        self.firm_rules = firm_rules
        self.tier = firm_rules.get_tier(tier_name)
        self.contract = contract
        self.commission_per_rt = commission_per_rt
        self.slippage_ticks = slippage_ticks
        self.reward_shaper = reward_shaper or RewardShaper()
        self.window = window

        obs_dim = observation_size(window)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Runtime state (reset on each episode)
        self._state: Optional[AccountState] = None
        self._broker: Optional[SimulatedBroker] = None
        self._monitor: Optional[RuleMonitor] = None
        self._bar_index: int = 0
        self._history: list[BarData] = []
        self._prev_equity: float = 0.0
        self._prev_date = None

    # ── Gymnasium interface ────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._state = AccountState(
            tier=self.tier,
            firm_rules=self.firm_rules,
            contract=self.contract,
        )
        self._broker = SimulatedBroker(
            commission_per_rt=self.commission_per_rt,
            slippage_ticks=self.slippage_ticks,
        )
        self._monitor = RuleMonitor()
        self._bar_index = self.window   # start after enough bars for observation
        self._history = list(self.bars[: self.window])
        self._prev_equity = self._state.equity
        self._prev_date = self.bars[self.window - 1].date if self.bars else None

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one bar.

        Returns (obs, reward, terminated, truncated, info).
        """
        if self._bar_index >= len(self.bars):
            obs = self._get_obs()
            return obs, 0.0, False, True, {}

        bar = self.bars[self._bar_index]
        state = self._state
        broker = self._broker
        monitor = self._monitor

        self._prev_equity = state.equity

        # ── EOD check ─────────────────────────────────────────────────
        current_date = bar.date
        if self._prev_date is not None and current_date != self._prev_date:
            monitor.update_eod(state)
        state.current_date = current_date
        self._prev_date = current_date

        # ── Execute any pending fill at bar open ───────────────────────
        fill = broker.process_pending(bar, self._bar_index, state)
        if fill is not None:
            state.trading_days_active.add(current_date)

        # ── Convert action to signal and queue ─────────────────────────
        signal = self._action_to_signal(action, state)
        if signal.action != "hold":
            broker.submit(signal, self._bar_index)

        # ── Update open PnL and HWM ────────────────────────────────────
        monitor.update_open_pnl(state, bar)
        broker.update_excursion(bar)
        monitor.update_intraday_hwm(state)

        # ── Check violations ───────────────────────────────────────────
        terminated = monitor.check_violations(state)

        # ── Reward ────────────────────────────────────────────────────
        reward = self.reward_shaper.compute(self._prev_equity, state, terminated)

        # ── Append bar and advance ─────────────────────────────────────
        self._history.append(bar)
        self._bar_index += 1

        truncated = (not terminated) and (self._bar_index >= len(self.bars))

        obs = self._get_obs()
        info = {
            "equity": state.equity,
            "drawdown_floor": state.drawdown_floor,
            "position": state.position_contracts,
            "realized_balance": state.realized_balance,
            "terminated_reason": self._terminated_reason(state) if (terminated or truncated) else "",
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        if len(self._history) < self.window:
            return np.zeros(observation_size(self.window), dtype=np.float32)
        return build_observation(self._history, self._state, window=self.window)

    @staticmethod
    def _action_to_signal(action_idx: int, state: AccountState) -> Signal:
        if action_idx == 0:
            return Signal(action="hold")
        elif action_idx == 1:
            if state.position_contracts < 0:
                return Signal(action="close")
            return Signal(action="buy", contracts=1)
        elif action_idx == 2:
            if state.position_contracts > 0:
                return Signal(action="close")
            return Signal(action="sell", contracts=1)
        elif action_idx == 3:
            return Signal(action="close")
        return Signal(action="hold")

    @staticmethod
    def _terminated_reason(state: AccountState) -> str:
        if state.breached_trailing_dd:
            return "trailing_drawdown_breach"
        if state.breached_daily_loss:
            return "daily_loss_breach"
        if state.hit_profit_target:
            return "profit_target_hit"
        return "data_exhausted"
