from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baseline import ACTION_TO_POSITION


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class BasisTradingEnv:
    def __init__(
        self,
        feature_frame: pd.DataFrame,
        observation_columns: list[str],
        fee_rate_per_rebalance: float,
        risk_penalty: float,
    ) -> None:
        if feature_frame.empty:
            raise ValueError("feature_frame is empty")
        if "basis" not in feature_frame.columns:
            raise ValueError("feature_frame must contain 'basis'")
        self.df = feature_frame.reset_index(drop=True)
        self.observation_columns = observation_columns
        self.fee_rate_per_rebalance = fee_rate_per_rebalance
        self.risk_penalty = risk_penalty

        self.index = 0
        self.position = 0
        self.cumulative_reward = 0.0

    def reset(self, start_index: int = 0) -> np.ndarray:
        if start_index < 0 or start_index >= len(self.df) - 1:
            raise ValueError("start_index is out of range")
        self.index = start_index
        self.position = 0
        self.cumulative_reward = 0.0
        return self._observation()

    def step(self, action: int) -> StepResult:
        if self.index >= len(self.df) - 1:
            raise RuntimeError("Environment is done. Call reset().")
        if action not in ACTION_TO_POSITION:
            raise ValueError(f"Unknown action id: {action}")

        target_position = ACTION_TO_POSITION[action]
        row = self.df.iloc[self.index]
        next_row = self.df.iloc[self.index + 1]

        basis_change = float(next_row["basis"] - row["basis"])
        pnl = target_position * basis_change
        rebalance_cost = self.fee_rate_per_rebalance * abs(target_position - self.position)
        risk_cost = self.risk_penalty * abs(target_position) * abs(float(row["basis_zscore"]))
        reward = pnl - rebalance_cost - risk_cost

        self.position = target_position
        self.index += 1
        self.cumulative_reward += reward

        done = self.index >= len(self.df) - 1
        obs = np.zeros(len(self.observation_columns) + 1, dtype=np.float64) if done else self._observation()
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "index": self.index,
                "position": self.position,
                "basis_change": basis_change,
                "pnl": pnl,
                "rebalance_cost": rebalance_cost,
                "risk_cost": risk_cost,
                "cumulative_reward": self.cumulative_reward,
            },
        )

    def _observation(self) -> np.ndarray:
        row = self.df.iloc[self.index]
        core = row[self.observation_columns].to_numpy(dtype=np.float64)
        return np.concatenate([core, np.array([self.position], dtype=np.float64)])
