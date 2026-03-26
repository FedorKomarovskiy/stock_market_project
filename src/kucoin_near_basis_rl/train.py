from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .baseline import BaselinePolicy
from .config import AppConfig, load_config
from .env import BasisTradingEnv
from .features import FEATURE_COLUMNS, build_feature_frame
from .kucoin_api import KuCoinPublicDataClient
from .qlearning import (
    QLearningAgent,
    StateDiscretizer,
    build_quantile_bins,
    save_model_artifact,
    train_qlearning,
)


@dataclass
class TrainingArtifacts:
    agent: QLearningAgent
    discretizer: StateDiscretizer
    observation_columns: list[str]
    history: list[float]
    rows_used: int


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_baseline_positions(feature_frame: pd.DataFrame, cfg: AppConfig) -> list[int]:
    baseline = BaselinePolicy(
        enter_zscore=cfg.baseline.enter_zscore,
        exit_zscore=cfg.baseline.exit_zscore,
    )
    baseline_positions: list[int] = []
    current_position = 0
    for row in feature_frame.itertuples():
        current_position = baseline.decide_position(float(row.basis_zscore), current_position)
        baseline_positions.append(current_position)
    return baseline_positions


def train_agent_from_features(feature_frame: pd.DataFrame, cfg: AppConfig) -> TrainingArtifacts:
    if len(feature_frame) < max(cfg.features.zscore_window, cfg.features.volatility_window) + 10:
        raise RuntimeError("Not enough candles after feature engineering.")

    baseline_positions = build_baseline_positions(feature_frame, cfg) if cfg.rl.use_baseline_guidance else None
    env = BasisTradingEnv(
        feature_frame=feature_frame,
        observation_columns=FEATURE_COLUMNS,
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
    )
    bins = build_quantile_bins(feature_frame, FEATURE_COLUMNS, cfg.rl.quantile_bins)
    discretizer = StateDiscretizer(bin_edges=bins)
    agent = QLearningAgent(
        num_actions=3,
        alpha=cfg.rl.alpha,
        gamma=cfg.rl.gamma,
        seed=42,
    )
    print(f"Training RL: episodes={cfg.rl.episodes}, rows={len(feature_frame)}")
    if cfg.rl.use_baseline_guidance:
        print("Baseline guidance: enabled")
    else:
        print("Baseline guidance: disabled (pure RL)")

    history = train_qlearning(
        env=env,
        agent=agent,
        discretizer=discretizer,
        baseline_positions=baseline_positions,
        episodes=cfg.rl.episodes,
        epsilon_start=cfg.rl.epsilon_start,
        epsilon_end=cfg.rl.epsilon_end,
        epsilon_decay=cfg.rl.epsilon_decay,
        imitation_start=cfg.rl.imitation_start,
        imitation_end=cfg.rl.imitation_end,
        imitation_decay=cfg.rl.imitation_decay,
        baseline_bonus=cfg.rl.baseline_bonus,
        max_steps_per_episode=cfg.rl.max_steps_per_episode,
    )
    return TrainingArtifacts(
        agent=agent,
        discretizer=discretizer,
        observation_columns=list(FEATURE_COLUMNS),
        history=history,
        rows_used=int(len(feature_frame)),
    )


def run_training(
    config_path: str,
    model_out: str,
    source_csv: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    features_out: str | None = None,
    episodes_override: int | None = None,
) -> dict[str, float]:
    cfg = load_config(config_path)
    if episodes_override is not None:
        if episodes_override <= 0:
            raise ValueError("episodes_override must be > 0")
        cfg.rl.episodes = int(episodes_override)

    if source_csv:
        raw = pd.read_csv(source_csv, parse_dates=["timestamp"])
        if raw["timestamp"].dt.tz is None:
            raw["timestamp"] = raw["timestamp"].dt.tz_localize("UTC")
    else:
        data_client = KuCoinPublicDataClient(cfg.api)
        if start_iso and end_iso:
            start_dt = _parse_dt(start_iso)
            end_dt = _parse_dt(end_iso)
        else:
            start_dt, end_dt = data_client.utc_lookback(cfg.data.lookback_minutes)
        raw = data_client.fetch_merged_candles(cfg.data, start_dt=start_dt, end_dt=end_dt)

    feature_frame = build_feature_frame(raw, cfg.features)
    artifacts = train_agent_from_features(feature_frame, cfg)

    save_model_artifact(
        path=model_out,
        agent=artifacts.agent,
        discretizer=artifacts.discretizer,
        observation_columns=artifacts.observation_columns,
        metadata={
            "config_path": str(config_path),
            "rows_used": int(artifacts.rows_used),
            "train_start": str(feature_frame.iloc[0]["timestamp"]),
            "train_end": str(feature_frame.iloc[-1]["timestamp"]),
        },
    )

    if features_out:
        out = Path(features_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        feature_frame.to_csv(out, index=False)

    return {
        "episodes": float(cfg.rl.episodes),
        "rows": float(len(feature_frame)),
        "avg_reward_last_10": float(
            sum(artifacts.history[-10:]) / max(1, len(artifacts.history[-10:]))
        ),
        "best_reward": float(max(artifacts.history)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NEAR basis RL model on KuCoin spot/futures data.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--model-out", required=True, help="Where to save model JSON.")
    parser.add_argument("--source-csv", default=None, help="Optional local CSV instead of downloading from KuCoin.")
    parser.add_argument("--start", default=None, help="UTC ISO timestamp for history start.")
    parser.add_argument("--end", default=None, help="UTC ISO timestamp for history end.")
    parser.add_argument("--features-out", default=None, help="Optional output CSV with engineered features.")
    parser.add_argument("--episodes", type=int, default=None, help="Optional override for RL episodes.")
    args = parser.parse_args()

    metrics = run_training(
        config_path=args.config,
        model_out=args.model_out,
        source_csv=args.source_csv,
        start_iso=args.start,
        end_iso=args.end,
        features_out=args.features_out,
        episodes_override=args.episodes,
    )
    print("Training complete:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
