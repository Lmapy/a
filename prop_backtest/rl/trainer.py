"""PPO training loop using Stable-Baselines3."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from prop_backtest.contracts.specs import ContractSpec
from prop_backtest.data.loader import BarData
from prop_backtest.firms.base import FirmRules
from prop_backtest.rl.env import PropFirmEnv
from prop_backtest.rl.rewards import RewardShaper


def train(
    firm_rules: FirmRules,
    tier_name: str,
    contract: ContractSpec,
    bars: list[BarData],
    total_timesteps: int = 500_000,
    model_path: str | Path = "models/ppo_agent",
    reward_shaper: Optional[RewardShaper] = None,
    n_envs: int = 4,
    window: int = 20,
    commission_per_rt: float = 4.50,
    slippage_ticks: int = 0,
    eval_freq: int = 10_000,
    verbose: int = 1,
) -> "PPO":  # noqa: F821
    """Train a PPO agent to pass a prop firm challenge.

    Uses vectorised environments (SubprocVecEnv) for faster training.
    Saves the trained model to ``model_path``.

    Args:
        firm_rules: Prop firm rules.
        tier_name: Account tier, e.g. "100K".
        contract: Futures contract spec.
        bars: Training bar data.
        total_timesteps: Total environment steps to train for.
        model_path: Where to save the trained model (SB3 adds .zip).
        reward_shaper: Custom reward shaper, or None for defaults.
        n_envs: Number of parallel environments.
        window: Observation window (must match at inference time).
        commission_per_rt: Commission per round-turn.
        slippage_ticks: Slippage in ticks.
        eval_freq: Evaluate on a separate env every N steps.
        verbose: SB3 verbosity level (0=silent, 1=info).

    Returns:
        Trained PPO model.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3[extra]"
        ) from e

    env_kwargs = dict(
        bars=bars,
        firm_rules=firm_rules,
        tier_name=tier_name,
        contract=contract,
        commission_per_rt=commission_per_rt,
        slippage_ticks=slippage_ticks,
        reward_shaper=reward_shaper,
        window=window,
    )

    # Vectorised training environments
    vec_env = make_vec_env(
        PropFirmEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
    )

    # Separate evaluation environment
    eval_env = PropFirmEnv(**env_kwargs)

    # Save path
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if eval_freq > 0:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(save_path.parent / "best"),
            log_path=str(save_path.parent / "logs"),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=verbose,
        )
        callbacks.append(eval_cb)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # small entropy bonus to encourage exploration
        tensorboard_log=str(save_path.parent / "tb_logs"),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=(verbose > 0),
    )

    model.save(str(save_path))
    if verbose > 0:
        print(f"\nModel saved to {save_path}.zip")

    vec_env.close()
    eval_env.close()

    return model
