from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn


@dataclass
class MetaDQNArtifacts:
    q_network: nn.Module
    observation_columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    episode_rewards: list[float]


@dataclass
class MetaPPOArtifacts:
    model: PPO
    observation_columns: list[str]
    mean: np.ndarray
    std: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_actions: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_meta_dqn_agent(
    train_frame: pd.DataFrame,
    observation_columns: list[str],
    episodes: int = 45,
    gamma: float = 0.98,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    replay_capacity: int = 25_000,
    min_replay_size: int = 512,
    target_sync_interval: int = 250,
    max_steps_per_episode: int = 512,
    seed: int = 42,
) -> MetaDQNArtifacts:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    mean = train_frame[observation_columns].mean().to_numpy(dtype=np.float32)
    std = train_frame[observation_columns].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
    device = torch.device("cpu")

    q_network = DQNetwork(input_dim=len(observation_columns) + 1).to(device)
    target_network = DQNetwork(input_dim=len(observation_columns) + 1).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    replay = ReplayBuffer(capacity=replay_capacity)
    episode_rewards: list[float] = []
    max_start = max(0, len(train_frame) - max_steps_per_episode - 2)
    global_step = 0

    for episode in range(episodes):
        epsilon = max(0.04, 0.25 * (0.95 ** episode))
        start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        index = start_index
        current_position = 0
        episode_reward = 0.0

        while index < len(train_frame) - 1 and (index - start_index) < max_steps_per_episode:
            state = _build_observation(train_frame, observation_columns, mean, std, index, current_position)
            row = train_frame.iloc[index]
            if float(row.get("rl_regime_active_flag", 1.0)) <= 0.0:
                action = 0
            elif rng.random() < epsilon:
                action = int(rng.choice(np.array([0, 1, 2], dtype=int)))
            else:
                with torch.no_grad():
                    q_values = q_network(torch.from_numpy(state).unsqueeze(0).to(device))
                    action = int(torch.argmax(q_values, dim=1).item())

            reward, next_position = _compute_meta_reward(train_frame, index, current_position, action)
            next_state = _build_observation(train_frame, observation_columns, mean, std, index + 1, next_position)
            done = bool(index + 1 >= len(train_frame) - 1 or (index - start_index + 1) >= max_steps_per_episode)
            replay.add(state, action, reward, next_state, done)
            episode_reward += reward
            current_position = next_position
            index += 1
            global_step += 1

            if len(replay) >= min_replay_size:
                batch = replay.sample(batch_size=min(batch_size, len(replay)))
                _optimize_dqn(q_network, target_network, optimizer, batch, gamma, device)
                if global_step % target_sync_interval == 0:
                    target_network.load_state_dict(q_network.state_dict())

        episode_rewards.append(float(episode_reward))

    return MetaDQNArtifacts(
        q_network=q_network,
        observation_columns=list(observation_columns),
        mean=mean,
        std=std,
        episode_rewards=episode_rewards,
    )


def rollout_meta_dqn_positions(frame: pd.DataFrame, artifacts: MetaDQNArtifacts) -> pd.DataFrame:
    current_position = 0
    rows: list[dict[str, int | float]] = []
    for index in range(len(frame)):
        state = _build_observation(
            frame,
            artifacts.observation_columns,
            artifacts.mean,
            artifacts.std,
            index,
            current_position,
        )
        with torch.no_grad():
            q_values = artifacts.q_network(torch.from_numpy(state).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
        target_position = _meta_target_position(frame.iloc[index], current_position, action)
        current_position = target_position
        rows.append(
            {
                "action": action,
                "target_position": current_position,
                "q_max": float(torch.max(q_values).item()),
            }
        )
    return pd.DataFrame(rows, index=frame.index)


class MetaControllerGymEnv(gym.Env[np.ndarray, int]):
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
        reward, next_position = _compute_meta_reward(self.df, self.index, self.current_position, int(action))
        self.current_position = next_position
        self.index += 1
        self.steps_taken += 1
        terminated = self.index >= len(self.df) - 1
        truncated = self.steps_taken >= self.max_steps
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if (terminated or truncated) else self._observation()
        return obs, float(reward), terminated, truncated, {
            "position": self.current_position,
            "signal_direction": float(self.df.iloc[self.index - 1].get("meta_signal_direction", 0.0)),
        }

    def _observation(self) -> np.ndarray:
        return _build_observation(
            self.df,
            self.observation_columns,
            self.mean,
            self.std,
            self.index,
            self.current_position,
        )


def train_meta_ppo_agent(
    train_frame: pd.DataFrame,
    observation_columns: list[str],
    total_timesteps: int = 20_000,
    seed: int = 42,
) -> MetaPPOArtifacts:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    mean = train_frame[observation_columns].mean().to_numpy(dtype=np.float32)
    std = train_frame[observation_columns].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)

    def make_env() -> MetaControllerGymEnv:
        return MetaControllerGymEnv(
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
        ent_coef=0.004,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=int(total_timesteps), progress_bar=False)
    return MetaPPOArtifacts(
        model=model,
        observation_columns=list(observation_columns),
        mean=mean,
        std=std,
    )


def rollout_meta_ppo_positions(frame: pd.DataFrame, artifacts: MetaPPOArtifacts) -> pd.DataFrame:
    env = MetaControllerGymEnv(
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
        target_position = _meta_target_position(env.df.iloc[env.index], env.current_position, int(action))
        rows.append({"action": int(action), "target_position": int(target_position)})
        obs, _reward, terminated, truncated, _info = env.step(int(action))
    while len(rows) < len(frame):
        rows.append({"action": 0, "target_position": 0})
    return pd.DataFrame(rows[: len(frame)], index=frame.index)


def _optimize_dqn(
    q_network: DQNetwork,
    target_network: DQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    gamma: float,
    device: torch.device,
) -> None:
    states, actions, rewards, next_states, dones = batch
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    q_values = q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_network(next_states_t).max(dim=1).values
        targets = rewards_t + gamma * (1.0 - dones_t) * next_q_values
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _build_observation(
    frame: pd.DataFrame,
    observation_columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    index: int,
    current_position: int,
) -> np.ndarray:
    row = frame.iloc[index]
    core = row[observation_columns].to_numpy(dtype=np.float32)
    normalized = (core - mean) / std
    return np.concatenate([normalized, np.array([float(current_position)], dtype=np.float32)])


def _meta_target_position(row: pd.Series, current_position: int, action: int) -> int:
    regime_active = float(row.get("rl_regime_active_flag", 1.0)) > 0.0
    signal_direction = int(np.sign(float(row.get("meta_signal_direction", 0.0))))
    if not regime_active or signal_direction == 0:
        return 0
    if action == 0:
        return 0
    if action == 1:
        return signal_direction
    return int(current_position)


def _compute_meta_reward(frame: pd.DataFrame, index: int, current_position: int, action: int) -> tuple[float, int]:
    row = frame.iloc[index]
    next_row = frame.iloc[index + 1]
    target_position = _meta_target_position(row, current_position, action)
    basis_change = float(next_row["basis"] - row["basis"])
    pnl = target_position * basis_change
    rebalance_cost = float(row.get("fee_rate_per_rebalance", 0.0)) * abs(target_position - current_position)
    if rebalance_cost == 0.0:
        rebalance_cost = 0.0004 * abs(target_position - current_position)
    risk_cost = 0.00002 * abs(target_position) * abs(float(row["basis_zscore"]))
    reward = pnl - (0.25 * rebalance_cost) - risk_cost

    signal_direction = int(np.sign(float(row.get("meta_signal_direction", 0.0))))
    signal_strength = float(row.get("meta_signal_strength", 0.0))
    alignment = float(row.get("meta_signal_alignment_flag", 0.0))
    regime_active = float(row.get("rl_regime_active_flag", 1.0))
    if regime_active > 0.0 and signal_direction != 0:
        if action == 1 and target_position == signal_direction:
            reward += 0.00002 + min(signal_strength, 3.0) * 0.000005
        if action == 0 and alignment > 0.0:
            reward -= 0.00003
        if action == 2 and current_position == signal_direction:
            reward += 0.00001
    if regime_active <= 0.0 and target_position != 0:
        reward -= 0.0001
    return float(reward), int(target_position)


@dataclass
class BinaryMetaDQNArtifacts:
    q_network: nn.Module
    observation_columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    episode_rewards: list[float]


@dataclass
class BinaryMetaPPOArtifacts:
    model: PPO
    observation_columns: list[str]
    mean: np.ndarray
    std: np.ndarray


def train_binary_meta_dqn_agent(
    train_frame: pd.DataFrame,
    observation_columns: list[str],
    episodes: int = 45,
    gamma: float = 0.98,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    replay_capacity: int = 25_000,
    min_replay_size: int = 512,
    target_sync_interval: int = 250,
    max_steps_per_episode: int = 512,
    seed: int = 42,
) -> BinaryMetaDQNArtifacts:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    mean = train_frame[observation_columns].mean().to_numpy(dtype=np.float32)
    std = train_frame[observation_columns].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
    device = torch.device("cpu")

    q_network = DQNetwork(input_dim=len(observation_columns) + 1, num_actions=2).to(device)
    target_network = DQNetwork(input_dim=len(observation_columns) + 1, num_actions=2).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    replay = ReplayBuffer(capacity=replay_capacity)
    episode_rewards: list[float] = []
    max_start = max(0, len(train_frame) - max_steps_per_episode - 2)
    global_step = 0

    for episode in range(episodes):
        epsilon = max(0.03, 0.20 * (0.95 ** episode))
        start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        index = start_index
        current_position = 0
        episode_reward = 0.0

        while index < len(train_frame) - 1 and (index - start_index) < max_steps_per_episode:
            state = _build_observation(train_frame, observation_columns, mean, std, index, current_position)
            row = train_frame.iloc[index]
            if not _binary_signal_active(row):
                action = 0
            elif rng.random() < epsilon:
                action = int(rng.choice(np.array([0, 1], dtype=int)))
            else:
                with torch.no_grad():
                    q_values = q_network(torch.from_numpy(state).unsqueeze(0).to(device))
                    action = int(torch.argmax(q_values, dim=1).item())

            reward, next_position = _compute_binary_meta_reward(train_frame, index, current_position, action)
            next_state = _build_observation(train_frame, observation_columns, mean, std, index + 1, next_position)
            done = bool(index + 1 >= len(train_frame) - 1 or (index - start_index + 1) >= max_steps_per_episode)
            replay.add(state, action, reward, next_state, done)
            episode_reward += reward
            current_position = next_position
            index += 1
            global_step += 1

            if len(replay) >= min_replay_size:
                batch = replay.sample(batch_size=min(batch_size, len(replay)))
                _optimize_dqn(q_network, target_network, optimizer, batch, gamma, device)
                if global_step % target_sync_interval == 0:
                    target_network.load_state_dict(q_network.state_dict())

        episode_rewards.append(float(episode_reward))

    return BinaryMetaDQNArtifacts(
        q_network=q_network,
        observation_columns=list(observation_columns),
        mean=mean,
        std=std,
        episode_rewards=episode_rewards,
    )


def rollout_binary_meta_dqn_positions(frame: pd.DataFrame, artifacts: BinaryMetaDQNArtifacts) -> pd.DataFrame:
    current_position = 0
    rows: list[dict[str, int | float]] = []
    for index in range(len(frame)):
        state = _build_observation(
            frame,
            artifacts.observation_columns,
            artifacts.mean,
            artifacts.std,
            index,
            current_position,
        )
        with torch.no_grad():
            q_values = artifacts.q_network(torch.from_numpy(state).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
        target_position = _binary_meta_target_position(frame.iloc[index], action)
        current_position = target_position
        rows.append(
            {
                "action": action,
                "target_position": current_position,
                "q_max": float(torch.max(q_values).item()),
            }
        )
    return pd.DataFrame(rows, index=frame.index)


class BinaryMetaControllerGymEnv(gym.Env[np.ndarray, int]):
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
        self.action_space = spaces.Discrete(2)
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
        reward, next_position = _compute_binary_meta_reward(self.df, self.index, self.current_position, int(action))
        self.current_position = next_position
        self.index += 1
        self.steps_taken += 1
        terminated = self.index >= len(self.df) - 1
        truncated = self.steps_taken >= self.max_steps
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if (terminated or truncated) else self._observation()
        return obs, float(reward), terminated, truncated, {
            "position": self.current_position,
            "signal_direction": float(self.df.iloc[self.index - 1].get("meta_signal_direction", 0.0)),
        }

    def _observation(self) -> np.ndarray:
        return _build_observation(
            self.df,
            self.observation_columns,
            self.mean,
            self.std,
            self.index,
            self.current_position,
        )


def train_binary_meta_ppo_agent(
    train_frame: pd.DataFrame,
    observation_columns: list[str],
    total_timesteps: int = 20_000,
    seed: int = 42,
) -> BinaryMetaPPOArtifacts:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    mean = train_frame[observation_columns].mean().to_numpy(dtype=np.float32)
    std = train_frame[observation_columns].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)

    def make_env() -> BinaryMetaControllerGymEnv:
        return BinaryMetaControllerGymEnv(
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
        ent_coef=0.002,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=int(total_timesteps), progress_bar=False)
    return BinaryMetaPPOArtifacts(
        model=model,
        observation_columns=list(observation_columns),
        mean=mean,
        std=std,
    )


def rollout_binary_meta_ppo_positions(frame: pd.DataFrame, artifacts: BinaryMetaPPOArtifacts) -> pd.DataFrame:
    env = BinaryMetaControllerGymEnv(
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
        target_position = _binary_meta_target_position(env.df.iloc[env.index], int(action))
        rows.append({"action": int(action), "target_position": int(target_position)})
        obs, _reward, terminated, truncated, _info = env.step(int(action))
    while len(rows) < len(frame):
        rows.append({"action": 0, "target_position": 0})
    return pd.DataFrame(rows[: len(frame)], index=frame.index)


def _binary_meta_target_position(row: pd.Series, action: int) -> int:
    signal_direction = int(np.sign(float(row.get("meta_signal_direction", 0.0))))
    if not _binary_signal_active(row) or signal_direction == 0:
        return 0
    if action == 0:
        return 0
    return signal_direction


def _compute_binary_meta_reward(frame: pd.DataFrame, index: int, current_position: int, action: int) -> tuple[float, int]:
    row = frame.iloc[index]
    next_row = frame.iloc[index + 1]
    target_position = _binary_meta_target_position(row, action)
    basis_change = float(next_row["basis"] - row["basis"])
    pnl = target_position * basis_change
    rebalance_cost = float(row.get("fee_rate_per_rebalance", 0.0)) * abs(target_position - current_position)
    if rebalance_cost == 0.0:
        rebalance_cost = 0.0004 * abs(target_position - current_position)
    risk_cost = 0.000015 * abs(target_position) * abs(float(row["basis_zscore"]))
    reward = pnl - (0.15 * rebalance_cost) - risk_cost

    regime_active = float(row.get("rl_regime_active_flag", 1.0))
    signal_direction = int(np.sign(float(row.get("meta_signal_direction", 0.0))))
    signal_strength = float(row.get("meta_signal_strength", 0.0))
    alignment = float(row.get("meta_signal_alignment_flag", 0.0))
    if _binary_signal_active(row) and regime_active > 0.0 and signal_direction != 0:
        if action == 1:
            reward += 0.000012 + min(signal_strength, 4.0) * 0.000004
            if alignment > 0.0:
                reward += 0.00002
        else:
            reward -= 0.000006
            if alignment > 0.0 and signal_strength >= 1.0:
                reward -= 0.000025
    elif action == 1:
        reward -= 0.00002
    return float(reward), int(target_position)


def _binary_signal_active(row: pd.Series) -> bool:
    regime_active = float(row.get("rl_regime_active_flag", 1.0)) > 0.0
    signal_direction = int(np.sign(float(row.get("meta_signal_direction", 0.0))))
    signal_strength = float(row.get("meta_signal_strength", 0.0))
    alignment = float(row.get("meta_signal_alignment_flag", 0.0))
    gb_confidence = float(row.get("gb_confidence_score", 0.0))
    return bool(
        regime_active
        and signal_direction != 0
        and (
            signal_strength >= 1.15
            or alignment > 0.0
            or gb_confidence >= 1.25
        )
    )
