"""RLStrategy — wraps a trained Stable-Baselines3 model as a Strategy."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from prop_backtest.account.state import AccountState
from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.engine.broker import Signal
from prop_backtest.firms.base import AccountTier
from prop_backtest.strategy.base import Strategy


class RLStrategy(Strategy):
    """Uses a trained PPO/SAC model (Stable-Baselines3) to generate signals.

    The model must have been trained with ``prop_backtest.rl.trainer.train()``
    or any environment that shares the same observation space defined in
    ``prop_backtest.rl.features.build_observation()``.

    Args:
        model_path: Path to the saved SB3 model file (without .zip extension).
        window: Number of historical bars to include in the observation.
                Must match the window used during training.
        deterministic: If True, use argmax policy (no exploration).
    """

    def __init__(
        self,
        model_path: str | Path,
        window: int = 20,
        deterministic: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.window = window
        self.deterministic = deterministic
        self._model = None

    def on_start(self, contract: ContractSpec, tier: AccountTier) -> None:
        self._load_model()

    def on_bar(
        self,
        bar: BarData,
        history: list[BarData],
        account: AccountState,
    ) -> Signal:
        from prop_backtest.rl.features import build_observation

        if len(history) < self.window:
            return Signal(action="hold")

        obs = build_observation(history, account, window=self.window)
        obs = obs.reshape(1, -1)  # SB3 expects (1, obs_dim)

        action_idx, _ = self._model.predict(obs, deterministic=self.deterministic)
        action_idx = int(action_idx)

        return self._action_to_signal(action_idx, account)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            ) from e

        if not self.model_path.exists() and not Path(str(self.model_path) + ".zip").exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._model = PPO.load(str(self.model_path))

    @staticmethod
    def _action_to_signal(action_idx: int, account: AccountState) -> Signal:
        """Map discrete action index to a trading Signal.

        0 = hold
        1 = buy 1 contract
        2 = sell 1 contract
        3 = close position
        """
        if action_idx == 0:
            return Signal(action="hold")
        elif action_idx == 1:
            if account.position_contracts < 0:
                return Signal(action="close")
            return Signal(action="buy", contracts=1)
        elif action_idx == 2:
            if account.position_contracts > 0:
                return Signal(action="close")
            return Signal(action="sell", contracts=1)
        elif action_idx == 3:
            return Signal(action="close")
        return Signal(action="hold")
