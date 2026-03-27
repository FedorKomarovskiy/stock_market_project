from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baseline import ACTION_TO_POSITION


@dataclass
class DQNArtifacts:
    q_network: nn.Module
    observation_columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    episode_rewards: list[float]


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


def train_dqn_agent(
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
) -> DQNArtifacts:
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
    global_step = 0

    max_start = max(0, len(train_frame) - max_steps_per_episode - 2)
    for episode in range(episodes):
        epsilon = max(0.05, 0.30 * (0.95 ** episode))
        start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        index = start_index
        current_position = 0
        episode_reward = 0.0

        while index < len(train_frame) - 1 and (index - start_index) < max_steps_per_episode:
            state = _build_observation(train_frame, observation_columns, mean, std, index, current_position)
            gate_active = float(train_frame.iloc[index].get("rl_regime_active_flag", 1.0)) > 0.0
            if not gate_active:
                action = 1
            elif rng.random() < epsilon:
                exploratory_actions = np.array([0, 1, 2], dtype=int)
                gb_direction = int(np.sign(float(train_frame.iloc[index].get("gb_pred_basis_change", 0.0))))
                if gb_direction > 0:
                    exploratory_actions = np.array([1, 2], dtype=int)
                elif gb_direction < 0:
                    exploratory_actions = np.array([0, 1], dtype=int)
                action = int(rng.choice(exploratory_actions))
            else:
                with torch.no_grad():
                    q_values = q_network(torch.from_numpy(state).unsqueeze(0).to(device))
                    action = int(torch.argmax(q_values, dim=1).item())
                action = _apply_gate_to_action(train_frame.iloc[index], action)

            reward, next_position = _compute_reward(train_frame, index, current_position, action)
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

    return DQNArtifacts(
        q_network=q_network,
        observation_columns=list(observation_columns),
        mean=mean,
        std=std,
        episode_rewards=episode_rewards,
    )


def rollout_dqn_positions(
    frame: pd.DataFrame,
    artifacts: DQNArtifacts,
) -> pd.DataFrame:
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
        action = _apply_gate_to_action(frame.iloc[index], action)
        current_position = ACTION_TO_POSITION[action]
        rows.append(
            {
                "action": action,
                "target_position": current_position,
                "q_max": float(torch.max(q_values).item()),
            }
        )
    return pd.DataFrame(rows, index=frame.index)


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


def _compute_reward(frame: pd.DataFrame, index: int, current_position: int, action: int) -> tuple[float, int]:
    row = frame.iloc[index]
    next_row = frame.iloc[index + 1]
    target_position = ACTION_TO_POSITION[action]
    basis_change = float(next_row["basis"] - row["basis"])
    pnl = target_position * basis_change
    rebalance_cost = float(row.get("fee_rate_per_rebalance", 0.0)) * abs(target_position - current_position)
    if rebalance_cost == 0.0:
        rebalance_cost = 0.0004 * abs(target_position - current_position)
    risk_cost = 0.00003 * abs(target_position) * abs(float(row["basis_zscore"]))
    reward = pnl - (0.35 * rebalance_cost) - risk_cost

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
    return float(reward), int(target_position)


def _apply_gate_to_action(row: pd.Series, action: int) -> int:
    if float(row.get("rl_regime_active_flag", 1.0)) <= 0.0:
        return 1
    gb_direction = int(np.sign(float(row.get("gb_pred_basis_change", 0.0))))
    baseline_position = int(row.get("baseline_position", 0))
    if gb_direction != 0 and baseline_position != 0 and gb_direction == baseline_position:
        forced = {-1: 0, 0: 1, 1: 2}.get(gb_direction, 1)
        if forced in (0, 2) and action == 1:
            return action
    if float(row.get("gb_high_confidence_flag", 0.0)) > 0.0 and gb_direction != 0:
        if gb_direction > 0 and action == 0:
            return 1
        if gb_direction < 0 and action == 2:
            return 1
    return action
