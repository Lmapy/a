"""Reward shaping for the prop firm RL environment."""
from __future__ import annotations

from prop_backtest.account.state import AccountState


class RewardShaper:
    """Computes step rewards for the RL agent.

    Reward components:
    1. base_reward   — normalised equity change this step.
    2. proximity_penalty — quadratic penalty when near the drawdown floor.
    3. termination_reward — large bonus/penalty when episode ends.

    All rewards are normalised by starting_balance so they are
    comparable across different account sizes.

    Args:
        alpha: Weight of the proximity penalty (default 0.5).
        pass_bonus: Reward added when the profit target is reached (default 2.0).
        fail_penalty: Reward added (negative) when a rule is violated (default -1.0).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        pass_bonus: float = 2.0,
        fail_penalty: float = -1.0,
    ) -> None:
        self.alpha = alpha
        self.pass_bonus = pass_bonus
        self.fail_penalty = fail_penalty

    def compute(
        self,
        prev_equity: float,
        state: AccountState,
        terminated: bool,
    ) -> float:
        """Compute reward for the current step.

        Args:
            prev_equity: Equity value at the start of this step.
            state: Current AccountState (after the step).
            terminated: Whether the episode ended this step.

        Returns:
            float reward.
        """
        start = state.starting_balance

        # ── 1. Base reward: normalised equity delta ────────────────────
        base = (state.equity - prev_equity) / start

        # ── 2. Proximity penalty: quadratic near the floor ─────────────
        proximity = state.dd_floor_proximity    # 0 = at floor, 1 = safe
        if proximity < 1.0:
            penalty = -self.alpha * (1.0 - max(proximity, 0.0)) ** 2
        else:
            penalty = 0.0

        # ── 3. Termination reward ──────────────────────────────────────
        term_reward = 0.0
        if terminated:
            if state.hit_profit_target:
                term_reward = self.pass_bonus
            elif state.breached_trailing_dd or state.breached_daily_loss:
                term_reward = self.fail_penalty

        return base + penalty + term_reward
