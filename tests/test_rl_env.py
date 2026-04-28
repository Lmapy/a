"""Tests for the PropFirmEnv Gymnasium environment."""
import pytest
from datetime import datetime, timedelta

import numpy as np

from prop_backtest.contracts.specs import get_contract
from prop_backtest.data.loader import BarData
from prop_backtest.firms import TOPSTEP_RULES
from prop_backtest.rl.env import PropFirmEnv
from prop_backtest.rl.features import build_observation, observation_size


ES = get_contract("ES")


def make_bars(n: int = 100, start_price: float = 5000.0) -> list[BarData]:
    bars = []
    base = datetime(2024, 1, 2, 9, 30)
    for i in range(n):
        close = start_price + i * 0.5
        bars.append(BarData(
            timestamp=base + timedelta(days=i),
            open=close,
            high=close + 1.0,
            low=close - 1.0,
            close=close,
            volume=500,
            contract=ES,
        ))
    return bars


def make_env(n_bars: int = 100, window: int = 20) -> PropFirmEnv:
    return PropFirmEnv(
        bars=make_bars(n_bars),
        firm_rules=TOPSTEP_RULES,
        tier_name="50K",
        contract=ES,
        commission_per_rt=0.0,
        window=window,
    )


# ── Observation space ─────────────────────────────────────────────────────────

def test_observation_size():
    window = 20
    expected = window * 5 + window + 6 + 2
    assert observation_size(window) == expected


def test_obs_shape_after_reset():
    env = make_env(window=20)
    obs, info = env.reset()
    assert obs.shape == (observation_size(20),)
    assert obs.dtype == np.float32


def test_action_space():
    env = make_env()
    assert env.action_space.n == 4


def test_observation_space_shape():
    env = make_env(window=20)
    assert env.observation_space.shape == (observation_size(20),)


# ── Step behaviour ────────────────────────────────────────────────────────────

def test_step_returns_correct_types():
    env = make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # hold
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_ends_on_truncation():
    """Running through all bars should eventually produce truncated=True."""
    env = make_env(n_bars=50, window=10)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    assert done, "Episode should end within 200 steps"


def test_reset_resets_state():
    env = make_env()
    obs1, _ = env.reset()
    env.step(1)
    env.step(1)
    obs2, _ = env.reset()
    # After reset, equity should be back to starting balance
    assert env._state.equity == env._state.starting_balance
    assert env._state.position_contracts == 0


def test_info_contains_equity():
    env = make_env()
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "equity" in info
    assert "position" in info
    assert "drawdown_floor" in info


# ── Reward shaping ────────────────────────────────────────────────────────────

def test_reward_is_finite():
    env = make_env()
    env.reset()
    for action in range(4):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward), f"Reward for action {action} is not finite: {reward}"
