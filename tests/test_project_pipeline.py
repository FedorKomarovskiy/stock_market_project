from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from kucoin_near_basis_rl.backtest import (
    apply_kelly_overlay,
    backtest_positions,
    build_baseline_positions,
    estimate_kelly_fraction,
    rollout_rl_positions,
)
from kucoin_near_basis_rl.config import AppConfig, FeatureConfig
from kucoin_near_basis_rl.features import FEATURE_COLUMNS, build_feature_frame
from kucoin_near_basis_rl.qlearning import QLearningAgent, StateDiscretizer, build_quantile_bins
from kucoin_near_basis_rl.train import train_agent_from_features


def _synthetic_raw(rows: int = 900) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = [start + timedelta(hours=i) for i in range(rows)]
    x = np.linspace(0.0, 30.0, rows)
    spot = 5.0 + 0.05 * np.sin(x) + 0.008 * np.cos(2.0 * x)
    basis = 0.003 * np.sin(0.7 * x) + 0.001 * np.cos(0.15 * x)
    futures = spot * (1.0 + basis)
    spot_volume = 10_000 + 800 * np.sin(0.3 * x)
    futures_volume = 10_500 + 700 * np.cos(0.25 * x)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "spot_close": spot,
            "futures_close": futures,
            "spot_volume": spot_volume,
            "futures_volume": futures_volume,
        }
    )


def test_rollout_rl_positions_defaults_to_flat_for_unknown_states() -> None:
    frame = build_feature_frame(_synthetic_raw(rows=400), FeatureConfig())
    bins = build_quantile_bins(frame, FEATURE_COLUMNS, [0.1, 0.5, 0.9])
    discretizer = StateDiscretizer(bin_edges=bins)
    untrained_agent = QLearningAgent(num_actions=3, alpha=0.1, gamma=0.98, seed=7)

    positions = rollout_rl_positions(
        feature_frame=frame.head(25),
        agent=untrained_agent,
        discretizer=discretizer,
        observation_columns=FEATURE_COLUMNS,
    )

    assert positions["state_known"].sum() == 0
    assert set(positions["target_position"].tolist()) == {0}
    assert set(positions["action"].tolist()) == {1}


def test_research_backtest_pipeline_on_synthetic_data() -> None:
    cfg = AppConfig()
    cfg.rl.episodes = 6
    cfg.rl.use_baseline_guidance = True
    cfg.rl.max_steps_per_episode = 300
    cfg.research.initial_capital = 10_000.0

    frame = build_feature_frame(_synthetic_raw(), cfg.features)
    split_index = int(len(frame) * 0.7)
    train_frame = frame.iloc[:split_index].reset_index(drop=True)
    test_frame = frame.iloc[split_index:].reset_index(drop=True)

    artifacts = train_agent_from_features(train_frame, cfg)
    baseline_positions = build_baseline_positions(
        test_frame,
        enter_zscore=cfg.baseline.enter_zscore,
        exit_zscore=cfg.baseline.exit_zscore,
    )
    rl_positions = rollout_rl_positions(
        feature_frame=test_frame,
        agent=artifacts.agent,
        discretizer=artifacts.discretizer,
        observation_columns=artifacts.observation_columns,
    )

    baseline_result = backtest_positions(
        feature_frame=test_frame,
        positions=baseline_positions.tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="baseline_zscore",
    )
    rl_result = backtest_positions(
        feature_frame=test_frame,
        positions=rl_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="rl_agent",
    )

    assert len(baseline_result.history) == len(test_frame) - 1
    assert len(rl_result.history) == len(test_frame) - 1
    for key in ("cagr", "sharpe", "max_drawdown", "final_equity"):
        assert key in baseline_result.metrics
        assert key in rl_result.metrics
        assert np.isfinite(float(baseline_result.metrics[key]))
        assert np.isfinite(float(rl_result.metrics[key]))


def test_meta_controller_rollout_stays_in_valid_positions() -> None:
    from kucoin_near_basis_rl.config import AppConfig
    from kucoin_near_basis_rl.feature_pipeline import RL_META_OBSERVATION_COLUMNS, prepare_frames
    from kucoin_near_basis_rl.meta_controller_agent import (
        rollout_binary_meta_dqn_positions,
        rollout_meta_dqn_positions,
        train_binary_meta_dqn_agent,
        train_meta_dqn_agent,
    )

    cfg = AppConfig()
    cfg.research.min_train_rows = 200
    cfg.research.min_test_rows = 100
    prepared = prepare_frames(_synthetic_raw(rows=800), cfg)
    artifacts = train_meta_dqn_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_META_OBSERVATION_COLUMNS,
        episodes=2,
        min_replay_size=64,
        max_steps_per_episode=128,
    )
    positions = rollout_meta_dqn_positions(prepared.test_frame.head(50), artifacts)

    assert set(positions["target_position"].tolist()).issubset({-1, 0, 1})

    binary_artifacts = train_binary_meta_dqn_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_META_OBSERVATION_COLUMNS,
        episodes=2,
        min_replay_size=64,
        max_steps_per_episode=128,
    )
    binary_positions = rollout_binary_meta_dqn_positions(prepared.test_frame.head(50), binary_artifacts)

    assert set(binary_positions["target_position"].tolist()).issubset({-1, 0, 1})


def test_kelly_overlay_supports_fractional_positions() -> None:
    frame = build_feature_frame(_synthetic_raw(rows=500), FeatureConfig())
    positions = [1.0 if i % 10 < 5 else 0.0 for i in range(len(frame))]
    kelly_fraction = estimate_kelly_fraction(
        feature_frame=frame,
        positions=positions,
        fee_rate_per_rebalance=0.0001,
        risk_penalty=0.00001,
        max_fraction=2.0,
        fraction_multiplier=0.5,
        min_active_rows=10,
    )
    scaled_positions = apply_kelly_overlay(positions, kelly_fraction)
    result = backtest_positions(
        feature_frame=frame,
        positions=scaled_positions,
        fee_rate_per_rebalance=0.0001,
        risk_penalty=0.00001,
        initial_capital=10_000.0,
        strategy_name="kelly_overlay_test",
    )

    assert 0.0 <= kelly_fraction <= 2.0
    assert len(result.history) == len(frame) - 1


def test_enhanced_meta_dqn_rollout_stays_in_valid_position_sizes() -> None:
    from kucoin_near_basis_rl.config import AppConfig
    from kucoin_near_basis_rl.enhanced_meta_dqn import (
        rollout_enhanced_meta_dqn_positions,
        train_enhanced_meta_dqn_agent,
    )
    from kucoin_near_basis_rl.feature_pipeline import RL_META_OBSERVATION_COLUMNS, add_gb_auxiliary_features, build_enriched_feature_frame

    cfg = AppConfig()
    cfg.research.min_train_rows = 200
    cfg.research.min_test_rows = 100
    feature_frame = build_enriched_feature_frame(_synthetic_raw(rows=900), cfg)
    split_index = 650
    train_frame = feature_frame.iloc[:split_index].reset_index(drop=True)
    test_frame = feature_frame.iloc[split_index:].reset_index(drop=True)
    train_prepared, test_prepared, _cols, _thr = add_gb_auxiliary_features(train_frame, test_frame, cfg)

    artifacts = train_enhanced_meta_dqn_agent(
        train_frame=train_prepared,
        observation_columns=RL_META_OBSERVATION_COLUMNS,
        episodes=2,
        min_replay_size=64,
        max_steps_per_episode=128,
        reward_horizon=3,
    )
    positions = rollout_enhanced_meta_dqn_positions(test_prepared.head(50), artifacts)

    assert set(positions["target_position"].tolist()).issubset({-1.0, -0.5, 0.0, 0.5, 1.0})
