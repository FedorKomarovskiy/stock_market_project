from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback for environments without gymnasium
    import gym
    from gym import spaces

from .positioning import build_base_neutral_position


class SimpleBasisEnv(gym.Env):
    """Simple basis-trading environment with discrete actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        capital_usd: float = 1000.0,
        contract_size: float = 1.0,
    ) -> None:
        super().__init__()
        required = {"timestamp", "spot", "perp", "basis", "basis_z", "basis_delta"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns for SimpleBasisEnv: {sorted(missing)}")
        if len(data) < 2:
            raise ValueError("SimpleBasisEnv requires at least 2 rows")

        self.data = data.sort_values("timestamp").reset_index(drop=True).copy()
        self.capital_usd = float(capital_usd)
        self.contract_size = float(contract_size)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        self._t = 0
        self.position = 0  # 0 flat, 1 long_basis, -1 short_basis
        self.spot_qty = 0.0
        self.perp_base_qty = 0.0
        self.spot_entry = 0.0
        self.perp_entry = 0.0
        self.equity = 0.0

    def _open_position(self, side: str, spot_price: float, perp_price: float) -> None:
        built = build_base_neutral_position(
            spot_price=spot_price,
            perp_price=perp_price,
            capital_usd=self.capital_usd,
            contract_size=self.contract_size,
            side=side,  # type: ignore[arg-type]
        )
        self.position = 1 if side == "long_basis" else -1
        self.spot_qty = float(built["spot_qty"])
        self.perp_base_qty = float(built["perp_base_qty"])
        self.spot_entry = spot_price
        self.perp_entry = perp_price

    def _close_position(self) -> None:
        self.position = 0
        self.spot_qty = 0.0
        self.perp_base_qty = 0.0
        self.spot_entry = 0.0
        self.perp_entry = 0.0

    def _unrealized_pnl(self, spot_price: float, perp_price: float) -> float:
        if self.position == 1:
            return self.spot_qty * (spot_price - self.spot_entry) - self.perp_base_qty * (
                perp_price - self.perp_entry
            )
        if self.position == -1:
            return -self.spot_qty * (spot_price - self.spot_entry) + self.perp_base_qty * (
                perp_price - self.perp_entry
            )
        return 0.0

    def _observation(self, idx: int) -> np.ndarray:
        row = self.data.iloc[idx]
        unrealized = self._unrealized_pnl(float(row["spot"]), float(row["perp"]))
        return np.array(
            [
                float(row["basis"]),
                float(row["basis_z"]),
                float(row["basis_delta"]),
                float(self.position),
                float(unrealized),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        self._close_position()
        self.equity = 0.0
        return self._observation(self._t), {}

    def step(self, action: int):
        if self._t >= len(self.data) - 1:
            raise RuntimeError("Episode is done. Call reset() before next step.")

        action = int(action)
        row_t = self.data.iloc[self._t]
        row_next = self.data.iloc[self._t + 1]

        spot_t = float(row_t["spot"])
        perp_t = float(row_t["perp"])
        spot_next = float(row_next["spot"])
        perp_next = float(row_next["perp"])

        if action == 1:
            self._open_position("long_basis", spot_price=spot_t, perp_price=perp_t)
        elif action == 2:
            self._open_position("short_basis", spot_price=spot_t, perp_price=perp_t)
        elif action == 3:
            self._close_position()
        elif action != 0:
            raise ValueError(f"Unsupported action: {action}")

        d_spot = spot_next - spot_t
        d_perp = perp_next - perp_t

        if self.position == 1:
            equity_change = self.spot_qty * d_spot - self.perp_base_qty * d_perp
        elif self.position == -1:
            equity_change = -self.spot_qty * d_spot + self.perp_base_qty * d_perp
        else:
            equity_change = 0.0

        self.equity += equity_change
        self._t += 1

        terminated = self._t >= len(self.data) - 1
        truncated = False
        next_obs = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._observation(self._t)
        )

        info = {
            "action": action,
            "basis": float(row_next["basis"]),
            "basis_z": float(row_next["basis_z"]),
            "position": int(self.position),
            "equity": float(self.equity),
        }
        return next_obs, float(equity_change), terminated, truncated, info

    def render(self) -> None:
        row = self.data.iloc[self._t]
        print(
            f"t={self._t} ts={row['timestamp']} basis={row['basis']:.6f} "
            f"z={row['basis_z']:.6f} position={self.position} equity={self.equity:.6f}"
        )

