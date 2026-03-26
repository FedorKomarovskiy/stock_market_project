from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from delta_bot.kucoin_client import KuCoinRestClient

from .data import prepare_market_dataframe
from .signals import add_basis_features

ACTION_NAMES = {
    0: "hold",
    1: "open_long_basis",
    2: "open_short_basis",
    3: "close",
}


def build_live_window_dataframe(
    *,
    client: KuCoinRestClient | None = None,
    symbol_spot: str = "NEAR-USDT",
    symbol_perp: str = "NEAR-USDTM",
    timeframe: str = "1m",
    lookback_bars: int = 500,
    rolling_window: int = 100,
) -> pd.DataFrame:
    market = prepare_market_dataframe(
        client=client,
        symbol_spot=symbol_spot,
        symbol_perp=symbol_perp,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
    )
    market = add_basis_features(market, window=rolling_window)
    return market.dropna().reset_index(drop=True)


def build_observation(
    *,
    market_df: pd.DataFrame,
    position: int,
    unrealized_pnl: float,
) -> np.ndarray:
    if market_df.empty:
        raise ValueError("market_df is empty")
    row = market_df.iloc[-1]
    return np.array(
        [
            float(row["basis"]),
            float(row["basis_z"]),
            float(row["basis_delta"]),
            float(position),
            float(unrealized_pnl),
        ],
        dtype=np.float32,
    )


def decide_action(
    *,
    model: Any,
    market_df: pd.DataFrame,
    position: int = 0,
    unrealized_pnl: float = 0.0,
) -> dict[str, Any]:
    obs = build_observation(
        market_df=market_df,
        position=position,
        unrealized_pnl=unrealized_pnl,
    )
    action, _ = model.predict(obs, deterministic=True)
    action_id = int(action)
    if action_id not in ACTION_NAMES:
        action_id = 0

    latest = market_df.iloc[-1]
    return {
        "action_id": action_id,
        "action_name": ACTION_NAMES[action_id],
        "basis": float(latest["basis"]),
        "basis_z": float(latest["basis_z"]),
    }

