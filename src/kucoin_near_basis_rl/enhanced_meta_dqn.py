from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn


STATEFUL_FEATURE_NAMES = [
    "current_position",
    "position_abs",
    "bars_in_trade_scaled",
    "entry_basis_zscore_scaled",
    "unrealized_pnl_scaled",
    "time_since_last_trade_scaled",
    "recent_turnover_rate",
]


@dataclass
class EnhancedMetaDQNArtifacts:
    q_network: nn.Module
    target_network: nn.Module
    observation_columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    episode_rewards: list[float]
    reward_horizon: int


@dataclass
class ControllerState:
    current_position: float = 0.0
    entry_basis: float = 0.0
    entry_basis_zscore: float = 0.0
    bars_in_trade: int = 0
    time_since_last_trade: int = 0
    recent_turnovers: deque[int] = field(default_factory=lambda: deque(maxlen=24))


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        max_priority = float(self.priorities.max()) if self.buffer else 1.0
        item = (state, int(action), float(reward), next_state, float(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.priorities[self.position] = max(max_priority, 1e-6)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        if not self.buffer:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        size = len(self.buffer)
        scaled = self.priorities[:size] ** self.alpha
        probs = scaled / np.maximum(scaled.sum(), 1e-12)
        indices = np.random.choice(size, size=batch_size, replace=size < batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        weights = (size * probs[indices]) ** (-beta)
        weights /= np.maximum(weights.max(), 1e-12)
        return (
            indices.astype(np.int64),
            (
                np.asarray(states, dtype=np.float32),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(next_states, dtype=np.float32),
                np.asarray(dones, dtype=np.float32),
            ),
            np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        clipped = np.maximum(np.asarray(priorities, dtype=np.float32), 1e-6)
        self.priorities[indices] = clipped

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_actions: int = 3) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def train_enhanced_meta_dqn_agent(
    train_frame: pd.DataFrame,
    observation_columns: list[str],
    episodes: int = 45,
    gamma: float = 0.98,
    learning_rate: float = 7e-4,
    batch_size: int = 96,
    replay_capacity: int = 40_000,
    min_replay_size: int = 768,
    target_sync_interval: int = 200,
    max_steps_per_episode: int = 768,
    reward_horizon: int = 6,
    imitation_start: float = 0.35,
    imitation_end: float = 0.10,
    seed: int = 42,
) -> EnhancedMetaDQNArtifacts:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    mean = train_frame[observation_columns].mean().to_numpy(dtype=np.float32)
    std = train_frame[observation_columns].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
    device = torch.device("cpu")

    input_dim = len(observation_columns) + len(STATEFUL_FEATURE_NAMES)
    q_network = DuelingDQNetwork(input_dim=input_dim, hidden_dim=160, num_actions=3).to(device)
    target_network = DuelingDQNetwork(input_dim=input_dim, hidden_dim=160, num_actions=3).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    replay = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=0.6)
    episode_rewards: list[float] = []
    max_start = max(0, len(train_frame) - max_steps_per_episode - reward_horizon - 2)
    global_step = 0

    for episode in range(episodes):
        epsilon = max(0.02, 0.18 * (0.96 ** episode))
        imitation_prob = max(
            float(imitation_end),
            float(imitation_start) * (0.96 ** episode),
        )
        beta = min(1.0, 0.4 + (0.6 * episode / max(1, episodes - 1)))
        start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        index = start_index
        state = ControllerState()
        episode_reward = 0.0

        while index < len(train_frame) - reward_horizon - 1 and (index - start_index) < max_steps_per_episode:
            observation = _build_observation(train_frame, observation_columns, mean, std, index, state)
            row = train_frame.iloc[index]
            if not _enhanced_signal_active(row):
                action = 0
            elif rng.random() < imitation_prob:
                action = _teacher_action(row)
            elif rng.random() < epsilon:
                action = int(rng.choice(np.array([0, 1, 2], dtype=int), p=np.array([0.15, 0.40, 0.45])))
            else:
                with torch.no_grad():
                    q_values = q_network(torch.from_numpy(observation).unsqueeze(0).to(device))
                    action = int(torch.argmax(q_values, dim=1).item())

            reward, next_state = _compute_enhanced_reward(
                frame=train_frame,
                index=index,
                state=state,
                action=action,
                reward_horizon=reward_horizon,
                gamma=gamma,
            )
            next_observation = _build_observation(train_frame, observation_columns, mean, std, index + 1, next_state)
            done = bool(index + reward_horizon + 1 >= len(train_frame) - 1 or (index - start_index + 1) >= max_steps_per_episode)
            replay.add(observation, action, reward, next_observation, done)
            episode_reward += reward
            state = next_state
            index += 1
            global_step += 1

            if len(replay) >= min_replay_size:
                indices, batch, weights = replay.sample(batch_size=min(batch_size, len(replay)), beta=beta)
                td_errors = _optimize_double_dqn(
                    q_network=q_network,
                    target_network=target_network,
                    optimizer=optimizer,
                    batch=batch,
                    weights=weights,
                    gamma=gamma,
                    device=device,
                )
                replay.update_priorities(indices, np.abs(td_errors) + 1e-6)
                if global_step % target_sync_interval == 0:
                    target_network.load_state_dict(q_network.state_dict())

        episode_rewards.append(float(episode_reward))

    return EnhancedMetaDQNArtifacts(
        q_network=q_network,
        target_network=target_network,
        observation_columns=list(observation_columns),
        mean=mean,
        std=std,
        episode_rewards=episode_rewards,
        reward_horizon=int(reward_horizon),
    )


def rollout_enhanced_meta_dqn_positions(
    frame: pd.DataFrame,
    artifacts: EnhancedMetaDQNArtifacts,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["action", "target_position", "q_max"], index=frame.index)
    state = ControllerState()
    rows: list[dict[str, float | int]] = []
    for index in range(len(frame)):
        observation = _build_observation(
            frame,
            artifacts.observation_columns,
            artifacts.mean,
            artifacts.std,
            index,
            state,
        )
        with torch.no_grad():
            q_values = artifacts.q_network(torch.from_numpy(observation).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
        target_position = _sized_target_position(frame.iloc[index], action)
        rows.append(
            {
                "action": action,
                "target_position": float(target_position),
                "q_max": float(torch.max(q_values).item()),
            }
        )
        state = _advance_state(frame.iloc[index], state, target_position)
    return pd.DataFrame(rows, index=frame.index)


def _optimize_double_dqn(
    q_network: DuelingDQNetwork,
    target_network: DuelingDQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    weights: np.ndarray,
    gamma: float,
    device: torch.device,
) -> np.ndarray:
    states, actions, rewards, next_states, dones = batch
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).to(device)
    weights_t = torch.from_numpy(weights).to(device)

    q_values = q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = q_network(next_states_t).argmax(dim=1)
        next_q_values = target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards_t + gamma * (1.0 - dones_t) * next_q_values

    td_errors = targets - q_values
    losses = nn.functional.smooth_l1_loss(q_values, targets, reduction="none")
    loss = torch.mean(losses * weights_t)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    optimizer.step()
    return td_errors.detach().cpu().numpy()


def _build_observation(
    frame: pd.DataFrame,
    observation_columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    index: int,
    state: ControllerState,
) -> np.ndarray:
    row = frame.iloc[index]
    core = row[observation_columns].to_numpy(dtype=np.float32)
    normalized = (core - mean) / std
    stateful = np.asarray(_stateful_features(row, state), dtype=np.float32)
    return np.concatenate([normalized, stateful])


def _stateful_features(row: pd.Series, state: ControllerState) -> list[float]:
    if abs(state.current_position) > 1e-9:
        unrealized = state.current_position * (float(row["basis"]) - float(state.entry_basis))
    else:
        unrealized = 0.0
    recent_turnover_rate = (
        float(sum(state.recent_turnovers)) / float(len(state.recent_turnovers))
        if state.recent_turnovers
        else 0.0
    )
    return [
        float(state.current_position),
        float(abs(state.current_position)),
        float(min(state.bars_in_trade / 24.0, 5.0)),
        float(np.clip(state.entry_basis_zscore / 5.0, -5.0, 5.0)),
        float(np.clip(unrealized / 0.01, -5.0, 5.0)),
        float(min(state.time_since_last_trade / 24.0, 5.0)),
        float(recent_turnover_rate),
    ]


def _sized_target_position(row: pd.Series, action: int) -> float:
    if not _enhanced_signal_active(row):
        return 0.0
    signal_direction = float(np.sign(float(row.get("meta_signal_direction", 0.0))))
    if signal_direction == 0.0:
        return 0.0
    if action <= 0:
        return 0.0
    if action == 1:
        return 0.75 * signal_direction
    return 1.0 * signal_direction


def _enhanced_signal_active(row: pd.Series) -> bool:
    baseline_extreme = max(
        float(row.get("baseline_enter_long_flag", 0.0)),
        float(row.get("baseline_enter_short_flag", 0.0)),
    ) > 0.0
    gb_confidence = float(row.get("gb_confidence_score", 0.0))
    signal_strength = float(row.get("meta_signal_strength", 0.0))
    return bool(
        float(row.get("rl_regime_active_flag", 0.0)) > 0.0
        and float(row.get("meta_signal_alignment_flag", 0.0)) > 0.0
        and (
            (gb_confidence >= 0.65 and signal_strength >= 0.80)
            or (baseline_extreme and gb_confidence >= 0.55 and signal_strength >= 0.75)
        )
        and int(np.sign(float(row.get("meta_signal_direction", 0.0)))) != 0
    )


def _advance_state(row: pd.Series, state: ControllerState, target_position: float) -> ControllerState:
    next_state = ControllerState(
        current_position=float(state.current_position),
        entry_basis=float(state.entry_basis),
        entry_basis_zscore=float(state.entry_basis_zscore),
        bars_in_trade=int(state.bars_in_trade),
        time_since_last_trade=int(state.time_since_last_trade),
        recent_turnovers=deque(state.recent_turnovers, maxlen=24),
    )
    turnover = abs(float(target_position) - float(state.current_position)) > 1e-9
    next_state.recent_turnovers.append(1 if turnover else 0)
    next_state.time_since_last_trade = 0 if turnover else int(state.time_since_last_trade) + 1

    same_direction = (
        abs(float(state.current_position)) > 1e-9
        and abs(float(target_position)) > 1e-9
        and np.sign(float(state.current_position)) == np.sign(float(target_position))
    )
    if abs(float(target_position)) <= 1e-9:
        next_state.current_position = 0.0
        next_state.entry_basis = float(row["basis"])
        next_state.entry_basis_zscore = 0.0
        next_state.bars_in_trade = 0
        return next_state

    if not same_direction:
        next_state.entry_basis = float(row["basis"])
        next_state.entry_basis_zscore = float(row["basis_zscore"])
        next_state.bars_in_trade = 1
    else:
        next_state.bars_in_trade = int(state.bars_in_trade) + 1
    next_state.current_position = float(target_position)
    return next_state


def _compute_enhanced_reward(
    frame: pd.DataFrame,
    index: int,
    state: ControllerState,
    action: int,
    reward_horizon: int,
    gamma: float,
) -> tuple[float, ControllerState]:
    row = frame.iloc[index]
    target_position = _sized_target_position(row, action)
    reward = 0.0
    max_h = min(int(reward_horizon), len(frame) - index - 1)
    for step in range(1, max_h + 1):
        prev_row = frame.iloc[index + step - 1]
        next_row = frame.iloc[index + step]
        basis_change = float(next_row["basis"] - prev_row["basis"])
        reward += (gamma ** (step - 1)) * float(target_position) * basis_change

    turnover = abs(float(target_position) - float(state.current_position))
    rebalance_cost = float(row.get("fee_rate_per_rebalance", 0.0)) * turnover
    if rebalance_cost == 0.0:
        rebalance_cost = 0.0004 * turnover
    reward -= 0.08 * rebalance_cost
    reward -= 0.00001 * abs(float(target_position)) * abs(float(row["basis_zscore"]))

    signal_strength = float(row.get("meta_signal_strength", 0.0))
    if _enhanced_signal_active(row):
        if action == 0:
            reward -= 0.00005 * min(signal_strength, 3.0)
        elif action == 1:
            reward += 0.000025 + (0.000007 * min(signal_strength, 3.0))
        else:
            reward += 0.000055 + (0.000012 * min(signal_strength, 3.0))
    elif action > 0:
        reward -= 0.00006

    next_state = _advance_state(row, state, target_position)
    return float(reward), next_state


def _teacher_action(row: pd.Series) -> int:
    if not _enhanced_signal_active(row):
        return 0
    signal_strength = float(row.get("meta_signal_strength", 0.0))
    gb_confidence = float(row.get("gb_confidence_score", 0.0))
    if signal_strength >= 1.20 or gb_confidence >= 1.05:
        return 2
    return 1
