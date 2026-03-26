from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .env import BasisTradingEnv


@dataclass
class StateDiscretizer:
    bin_edges: list[list[float]]
    _bin_edges_np: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bin_edges_np = [np.asarray(edges, dtype=np.float64) for edges in self.bin_edges]

    def transform(self, observation: np.ndarray) -> tuple[int, ...]:
        if len(observation) != len(self._bin_edges_np):
            raise ValueError("Observation size does not match number of bin sets.")
        state: list[int] = []
        obs = np.asarray(observation, dtype=np.float64)
        for value, edges in zip(obs, self._bin_edges_np):
            if np.isnan(value):
                state.append(0)
                continue
            if edges.size == 0:
                state.append(0)
                continue
            bucket = int(np.searchsorted(edges, value, side="left"))
            state.append(bucket)
        return tuple(state)


@dataclass
class QLearningAgent:
    num_actions: int
    alpha: float
    gamma: float
    seed: int = 42
    q_table: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def get_q_values(self, state: tuple[int, ...]) -> np.ndarray:
        key = self._key(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * self.num_actions
        return np.asarray(self.q_table[key], dtype=np.float64)

    def has_state(self, state: tuple[int, ...]) -> bool:
        return self._key(state) in self.q_table

    def greedy_action(self, state: tuple[int, ...]) -> int:
        q_values = self.get_q_values(state)
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def select_action(
        self,
        state: tuple[int, ...],
        epsilon: float,
        baseline_action: int | None = None,
        imitation_prob: float = 0.0,
    ) -> int:
        if baseline_action is not None and self.rng.random() < imitation_prob:
            return int(baseline_action)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return self.greedy_action(state)

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        q_values = self.get_q_values(state)
        next_q = self.get_q_values(next_state)
        td_target = reward if done else reward + self.gamma * float(next_q.max())
        q_values[action] += self.alpha * (td_target - q_values[action])
        self.q_table[self._key(state)] = q_values.tolist()

    @staticmethod
    def _key(state: tuple[int, ...]) -> str:
        return "|".join(str(x) for x in state)


def build_quantile_bins(
    feature_frame: pd.DataFrame,
    observation_columns: list[str],
    quantiles: list[float],
) -> list[list[float]]:
    if not quantiles:
        raise ValueError("quantiles cannot be empty")
    bins: list[list[float]] = []
    for column in observation_columns:
        values = feature_frame[column].to_numpy(dtype=np.float64)
        edges = np.quantile(values, quantiles).tolist()
        edges = sorted(set(float(x) for x in edges))
        bins.append(edges)
    # Position feature bins are deterministic for {-1, 0, 1}.
    bins.append([-0.5, 0.5])
    return bins


def train_qlearning(
    env: BasisTradingEnv,
    agent: QLearningAgent,
    discretizer: StateDiscretizer,
    baseline_positions: list[int] | None,
    episodes: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    imitation_start: float,
    imitation_end: float,
    imitation_decay: float,
    baseline_bonus: float,
    max_steps_per_episode: int,
) -> list[float]:
    epsilon = epsilon_start
    imitation_prob = imitation_start
    history: list[float] = []

    max_start = max(0, len(env.df) - max_steps_per_episode - 2)
    for _ in range(episodes):
        start_index = int(agent.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        obs = env.reset(start_index=start_index)
        state = discretizer.transform(obs)
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            baseline_action: int | None = None
            if baseline_positions is not None:
                baseline_position = int(baseline_positions[env.index])
                baseline_action = {-1: 0, 0: 1, 1: 2}[baseline_position]
            action = agent.select_action(
                state=state,
                epsilon=epsilon,
                baseline_action=baseline_action,
                imitation_prob=imitation_prob,
            )

            result = env.step(action)
            reward = result.reward
            if baseline_action is not None and action == baseline_action:
                reward += baseline_bonus
            next_state = discretizer.transform(result.observation)
            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=result.done,
            )
            state = next_state
            total_reward += reward
            done = result.done
            steps += 1

        history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        imitation_prob = max(imitation_end, imitation_prob * imitation_decay)
    return history


def save_model_artifact(
    path: str | Path,
    agent: QLearningAgent,
    discretizer: StateDiscretizer,
    observation_columns: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    artifact = {
        "agent": {
            "num_actions": agent.num_actions,
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "seed": agent.seed,
            "q_table": agent.q_table,
        },
        "discretizer": {"bin_edges": discretizer.bin_edges},
        "observation_columns": observation_columns,
        "metadata": metadata or {},
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def load_model_artifact(path: str | Path) -> tuple[QLearningAgent, StateDiscretizer, list[str], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    agent_data = payload["agent"]
    agent = QLearningAgent(
        num_actions=int(agent_data["num_actions"]),
        alpha=float(agent_data["alpha"]),
        gamma=float(agent_data["gamma"]),
        seed=int(agent_data.get("seed", 42)),
        q_table={k: list(v) for k, v in agent_data.get("q_table", {}).items()},
    )
    discretizer = StateDiscretizer(bin_edges=[[float(x) for x in edges] for edges in payload["discretizer"]["bin_edges"]])
    obs_cols = [str(x) for x in payload["observation_columns"]]
    metadata = payload.get("metadata", {})
    return agent, discretizer, obs_cols, metadata
