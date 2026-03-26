from __future__ import annotations

import math


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    if value == 0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * math.floor(abs(value) / step) * step


def ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    if value == 0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * math.ceil(abs(value) / step) * step


def bps(value_a: float, value_b: float) -> float:
    denom = max((value_a + value_b) / 2.0, 1e-12)
    return abs(value_a - value_b) / denom * 10000.0
