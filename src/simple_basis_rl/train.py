from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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


def _rollout_policy(model: PPO, data: pd.DataFrame, env: SimpleBasisEnv) -> pd.DataFrame:
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    t = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        row = data.iloc[t + 1]
        rows.append(
            {
                "timestamp": row["timestamp"],
                "spot": float(row["spot"]),
                "perp": float(row["perp"]),
                "basis": float(row["basis"]),
                "basis_z": float(row["basis_z"]),
                "action": int(action),
                "position": int(info["position"]),
                "reward": float(reward),
                "equity": float(info["equity"]),
            }
        )
        done = bool(terminated or truncated)
        t += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO model for simple basis trading")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--model-out", default="models/trained_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbol-spot", default="")
    parser.add_argument("--symbol-perp", default="")
    parser.add_argument("--timeframe", default="")
    parser.add_argument("--lookback-bars", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = _load_config(args.config)
    symbol_spot = args.symbol_spot or cfg.get("symbol_spot", "NEAR-USDT")
    symbol_perp = args.symbol_perp or cfg.get("symbol_perp", "NEAR-USDTM")
    timeframe = args.timeframe or cfg.get("timeframe", "1m")
    lookback_bars = int(args.lookback_bars or cfg.get("history_lookback_bars", 500))
    rolling_window = int(cfg.get("rolling_window", 100))
    capital_usd = float(cfg.get("capital_usd", 1000.0))
    contract_size = float(cfg.get("contract_size", 1.0))
    total_timesteps = int(cfg.get("train_timesteps", 50_000))

    market = prepare_market_dataframe(
        symbol_spot=symbol_spot,
        symbol_perp=symbol_perp,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
    )
    market = add_basis_features(market, window=rolling_window).dropna().reset_index(drop=True)
    if len(market) < 20:
        raise ValueError("Not enough rows after feature preparation")

    vec_env = DummyVecEnv(
        [lambda: SimpleBasisEnv(market, capital_usd=capital_usd, contract_size=contract_size)]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=args.seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))
    trained_model_zip = model_out.with_suffix(".zip")

    eval_env = SimpleBasisEnv(market, capital_usd=capital_usd, contract_size=contract_size)
    history = _rollout_policy(model=model, data=market, env=eval_env)

    total_pnl = float(history["equity"].iloc[-1]) if not history.empty else 0.0
    max_dd = _max_drawdown(history["equity"].to_numpy(dtype=float)) if not history.empty else 0.0
    trades = _number_of_trades(history["position"]) if not history.empty else 0

    metrics = pd.DataFrame(
        [
            {
                "total_pnl": total_pnl,
                "max_drawdown": max_dd,
                "number_of_trades": trades,
                "steps": int(len(history)),
            }
        ]
    )
    metrics_path = model_out.parent / "training_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    experiment_config = {
        "config_path": str(args.config),
        "seed": args.seed,
        "symbol_spot": symbol_spot,
        "symbol_perp": symbol_perp,
        "timeframe": timeframe,
        "lookback_bars": lookback_bars,
        "rolling_window": rolling_window,
        "capital_usd": capital_usd,
        "contract_size": contract_size,
        "train_timesteps": total_timesteps,
        "trained_model": str(trained_model_zip),
    }
    experiment_config_path = model_out.parent / "experiment_config.json"
    experiment_config_path.write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Model saved: {trained_model_zip}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Experiment config saved: {experiment_config_path}")


if __name__ == "__main__":
    main()

