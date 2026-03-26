from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

SignalName = Literal["open_long_basis", "open_short_basis", "close", "hold"]


def add_basis_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Add basis features to an aligned spot/perp frame."""

    required = {"timestamp", "spot", "perp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["spot"] = pd.to_numeric(out["spot"], errors="coerce")
    out["perp"] = pd.to_numeric(out["perp"], errors="coerce")
    out = out.dropna(subset=["spot", "perp"]).reset_index(drop=True)

    out["basis"] = out["perp"] - out["spot"]

    # Use only past observations to avoid look-ahead leakage.
    past_basis = out["basis"].shift(1)
    out["rolling_mean"] = past_basis.rolling(window=window, min_periods=window).mean()
    out["rolling_std"] = past_basis.rolling(window=window, min_periods=window).std(ddof=0)

    safe_std = out["rolling_std"].replace(0.0, np.nan)
    out["basis_z"] = (out["basis"] - out["rolling_mean"]) / safe_std
    out["basis_z"] = out["basis_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["basis_delta"] = out["basis"].diff().fillna(0.0)
    return out


def generate_signal(row: pd.Series) -> SignalName:
    """Generate strategy action label from one row with basis_z."""

    z = float(row.get("basis_z", 0.0))
    if z > 2.0:
        return "open_long_basis"
    if z < -2.0:
        return "open_short_basis"
    if abs(z) < 0.5:
        return "close"
    return "hold"

