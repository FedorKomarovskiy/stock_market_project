from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from .baseline import ACTION_TO_POSITION


@dataclass
class PPOArtifacts:
    model: PPO
    observation_columns: list[str]
    mean: np.ndarray
    std: np.ndarray


class BasisTradingGymEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        observation_columns: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        max_steps: int = 512,
        random_start: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.df = frame.reset_index(drop=True)
        self.observation_columns = list(observation_columns)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std == 0.0, 1.0, std).astype(np.float32)
        self.max_steps = int(max_steps)
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(seed)

        obs_dim = len(self.observation_columns) + 1
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.index = 0
        self.start_index = 0
        self.current_position = 0
        self.steps_taken = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = max(0, len(self.df) - self.max_steps - 2)
        self.start_index = int(self.rng.integers(0, max_start + 1)) if self.random_start and max_start > 0 else 0
        self.index = self.start_index
        self.current_position = 0
        self.steps_taken = 0
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        row = self.df.iloc[self.index]
        action = _apply_gate_to_action(row, int(action))
        target_position = ACTION_TO_POSITION[action]
        next_row = self.df.iloc[self.index + 1]
        basis_change = float(next_row["basis"] - row["basis"])
        pnl = target_position * basis_change
        rebalance_cost = 0.35 * 0.0004 * abs(target_position - self.current_position)
        risk_cost = 0.00003 * abs(target_position) * abs(float(row["basis_zscore"]))
        reward = pnl - rebalance_cost - risk_cost

        gb_direction = int(np.sign(float(row.get("gb_pred_basis_change", 0.0))))
        gb_confidence_flag = float(row.get("gb_high_confidence_flag", 0.0))
        baseline_position = int(row.get("baseline_position", 0))
        regime_active = float(row.get("rl_regime_active_flag", 1.0))
        if regime_active <= 0.0 and target_position != 0:
            reward -= 0.00012
        if gb_confidence_flag > 0 and target_position == gb_direction and target_position != 0:
            reward += 0.00002
        if gb_confidence_flag > 0 and gb_direction != 0 and target_position == -gb_direction:
            reward -= 0.00002
        if baseline_position != 0 and target_position == baseline_position:
            reward += 0.00002
        if baseline_position != 0 and target_position == -baseline_position:
            reward -= 0.00002

        self.current_position = target_position
        self.index += 1
        self.steps_taken += 1
        terminated = self.index >= len(self.df) - 1
        truncated = self.steps_taken >= self.max_steps
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if (terminated or truncated) else self._observation()
        return obs, float(reward), terminated, truncated, {
            "position": self.current_position,
            "basis_change": basis_change,
        }

    def _observation(self) -> np.ndarray:
        row = self.df.iloc[self.index]
        core = row[self.observation_columns].to_numpy(dtype=np.float32)
        normalized = (core - self.mean) / self.std
        return np.concatenate([normalized, np.array([float(self.current_position)], dtype=np.float32)])


def train_ppo_agent(
    train_frame: pd.DataFrame,
    observation_columns: list[str],
    total_timesteps: int = 20_000,
    seed: int = 42,
) -> PPOArtifacts:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    mean = train_frame[observation_columns].mean().to_numpy(dtype=np.float32)
    std = train_frame[observation_columns].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)

    def make_env() -> BasisTradingGymEnv:
        return BasisTradingGymEnv(
            frame=train_frame,
            observation_columns=observation_columns,
            mean=mean,
            std=std,
            max_steps=min(512, max(64, len(train_frame) - 2)),
            random_start=True,
            seed=seed,
        )

    vec_env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        seed=seed,
        n_steps=256,
        batch_size=64,
        gamma=0.98,
        learning_rate=3e-4,
        ent_coef=0.005,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=int(total_timesteps), progress_bar=False)
    return PPOArtifacts(
        model=model,
        observation_columns=list(observation_columns),
        mean=mean,
        std=std,
    )


def rollout_ppo_positions(frame: pd.DataFrame, artifacts: PPOArtifacts) -> pd.DataFrame:
    env = BasisTradingGymEnv(
        frame=frame,
        observation_columns=artifacts.observation_columns,
        mean=artifacts.mean,
        std=artifacts.std,
        max_steps=max(64, len(frame) - 2),
        random_start=False,
    )
    obs, _ = env.reset()
    rows: list[dict[str, int | float]] = []
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _ = artifacts.model.predict(obs, deterministic=True)
        row = env.df.iloc[env.index]
        gated_action = _apply_gate_to_action(row, int(action))
        target_position = ACTION_TO_POSITION[gated_action]
        rows.append({"action": int(gated_action), "target_position": int(target_position)})
        obs, _reward, terminated, truncated, _info = env.step(int(action))
    while len(rows) < len(frame):
        rows.append({"action": 1, "target_position": 0})
    return pd.DataFrame(rows[: len(frame)], index=frame.index)


def _apply_gate_to_action(row: pd.Series, action: int) -> int:
    if float(row.get("rl_regime_active_flag", 1.0)) <= 0.0:
        return 1
    gb_direction = int(np.sign(float(row.get("gb_pred_basis_change", 0.0))))
    if float(row.get("gb_high_confidence_flag", 0.0)) > 0.0 and gb_direction != 0:
        if gb_direction > 0 and action == 0:
            return 1
        if gb_direction < 0 and action == 2:
            return 1
    return int(action)
