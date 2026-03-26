from __future__ import annotations

from dataclasses import dataclass


ACTION_TO_POSITION = {
    0: -1,  # short basis: short futures + long spot
    1: 0,   # flat
    2: 1,   # long basis: long futures + short spot
}
POSITION_TO_ACTION = {v: k for k, v in ACTION_TO_POSITION.items()}


@dataclass
class BaselinePolicy:
    enter_zscore: float
    exit_zscore: float

    def decide_position(self, zscore: float, current_position: int) -> int:
        if zscore >= self.enter_zscore:
            return -1
        if zscore <= -self.enter_zscore:
            return 1
        if abs(zscore) <= self.exit_zscore:
            return 0
        return current_position
