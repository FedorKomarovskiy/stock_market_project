from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .data import prepare_market_dataframe
from .env import SimpleBasisEnv
from .signals import add_basis_features


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "simple_basis_rl.json"


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    return float(drawdown.min())


def _number_of_trades(position_series: pd.Series) -> int:
    prev = position_series.shift(1).fillna(0)
    opened = (prev == 0) & (position_series != 0)
    return int(opened.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained PPO policy on historical data")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-csv", default="reports/run_policy_history.csv")
    parser.add_argument("--symbol-spot", default="")
    parser.add_argument("--symbol-perp", default="")
    parser.add_argument("--timeframe", default="")
    parser.add_argument("--lookback-bars", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    symbol_spot = args.symbol_spot or cfg.get("symbol_spot", "NEAR-USDT")
    symbol_perp = args.symbol_perp or cfg.get("symbol_perp", "NEAR-USDTM")
    timeframe = args.timeframe or cfg.get("timeframe", "1m")
    lookback_bars = int(args.lookback_bars or cfg.get("history_lookback_bars", 500))
    rolling_window = int(cfg.get("rolling_window", 100))
    capital_usd = float(cfg.get("capital_usd", 1000.0))
    contract_size = float(cfg.get("contract_size", 1.0))
    model_path = args.model_path or cfg.get("rl_model_path", "models/ppo_basis_near.zip")

    model = PPO.load(model_path)

    market = prepare_market_dataframe(
        symbol_spot=symbol_spot,
        symbol_perp=symbol_perp,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
    )
    market = add_basis_features(market, window=rolling_window).dropna().reset_index(drop=True)
    if len(market) < 2:
        raise ValueError("Not enough rows for policy rollout")

    env = SimpleBasisEnv(market, capital_usd=capital_usd, contract_size=contract_size)
    obs, _ = env.reset()

    rows: list[dict[str, Any]] = []
    t = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        row = market.iloc[t + 1]
        rows.append(
            {
                "timestamp": row["timestamp"],
                "spot": float(row["spot"]),
                "perp": float(row["perp"]),
                "basis": float(row["basis"]),
                "basis_z": float(row["basis_z"]),
                "action": int(action),
                "position": int(info["position"]),
                "equity": float(info["equity"]),
            }
        )
        done = bool(terminated or truncated)
        t += 1

    out = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    total_pnl = float(out["equity"].iloc[-1]) if not out.empty else 0.0
    max_dd = _max_drawdown(out["equity"].to_numpy(dtype=float)) if not out.empty else 0.0
    num_trades = _number_of_trades(out["position"]) if not out.empty else 0

    print(f"Saved policy rollout: {output_path}")
    print(f"total_pnl={total_pnl:.6f}")
    print(f"max_drawdown={max_dd:.6f}")
    print(f"number_of_trades={num_trades}")


if __name__ == "__main__":
    main()

