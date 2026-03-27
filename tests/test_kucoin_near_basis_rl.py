from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from kucoin_near_basis_rl.baseline import BaselinePolicy
from kucoin_near_basis_rl.config import FeatureConfig
from kucoin_near_basis_rl.env import BasisTradingEnv
from kucoin_near_basis_rl.features import FEATURE_COLUMNS, build_feature_frame
from kucoin_near_basis_rl.forecast_strategy import build_price_model_frame, split_feature_frame
from kucoin_near_basis_rl.gb_model import train_gradient_boosting_models
from kucoin_near_basis_rl.kucoin_api import KuCoinExecutionClient
from kucoin_near_basis_rl.qlearning import (
    QLearningAgent,
    StateDiscretizer,
    build_quantile_bins,
    train_qlearning,
)


def _synthetic_raw(rows: int = 500) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(rows)]
    x = np.linspace(0.0, 30.0, rows)
    spot = 5.0 + 0.04 * np.sin(x) + 0.005 * np.cos(2.0 * x)
    basis = 0.002 * np.sin(0.5 * x)
    futures = spot * (1.0 + basis)
    spot_volume = 1000 + 120 * np.sin(0.7 * x)
    futures_volume = 1100 + 80 * np.cos(0.5 * x)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "spot_close": spot,
            "futures_close": futures,
            "spot_volume": spot_volume,
            "futures_volume": futures_volume,
        }
    )


def test_feature_frame_has_required_columns() -> None:
    raw = _synthetic_raw()
    frame = build_feature_frame(raw, FeatureConfig())
    assert not frame.empty
    for col in FEATURE_COLUMNS:
        assert col in frame.columns
    assert "basis_ema_gap" in frame.columns


def test_feature_frame_merges_news_features() -> None:
    raw = _synthetic_raw(rows=300)
    news = pd.DataFrame(
        {
            "timestamp": raw["timestamp"],
            "news_count_6h": 1.0,
            "news_sentiment_mean_6h": 0.2,
            "news_weighted_sentiment_mean_6h": 0.25,
            "news_sentiment_sum_6h": 0.2,
            "news_sentiment_abs_sum_6h": 0.3,
            "news_relevance_mean_6h": 0.5,
            "news_near_count_6h": 0.0,
            "news_positive_count_6h": 1.0,
            "news_negative_count_6h": 0.0,
            "news_cryptopanic_count_6h": 1.0,
            "news_finnhub_count_6h": 1.0,
            "news_x_count_6h": 0.0,
            "news_source_diversity_6h": 2.0,
            "news_count_24h": 2.0,
            "news_sentiment_mean_24h": 0.1,
            "news_weighted_sentiment_mean_24h": 0.15,
            "news_sentiment_sum_24h": 0.2,
            "news_sentiment_abs_sum_24h": 0.4,
            "news_relevance_mean_24h": 0.5,
            "news_near_count_24h": 1.0,
            "news_positive_count_24h": 2.0,
            "news_negative_count_24h": 0.0,
            "news_cryptopanic_count_24h": 2.0,
            "news_finnhub_count_24h": 1.0,
            "news_x_count_24h": 0.0,
            "news_source_diversity_24h": 2.0,
            "news_burst_ratio_6h_24h": 0.5,
        }
    )
    frame = build_feature_frame(raw, FeatureConfig(), news_feature_frame=news)
    assert "news_count_6h" in frame.columns
    assert float(frame.iloc[-1]["news_count_24h"]) == 2.0
    assert float(frame.iloc[-1]["news_finnhub_count_24h"]) == 1.0


def test_combined_sentiment_score_distinguishes_positive_and_negative() -> None:
    from kucoin_near_basis_rl.finnhub_news import _combined_sentiment_score

    positive = _combined_sentiment_score(
        "NEAR rallies after strong partnership and adoption news",
        "Token shows breakout momentum and inflow",
    )
    negative = _combined_sentiment_score(
        "NEAR drops after hack investigation and selloff fears",
        "Traders fear liquidation risk and outflow",
    )

    assert positive > 0.0
    assert negative < 0.0
    assert positive > negative


def test_gradient_boosting_training_runs() -> None:
    from kucoin_near_basis_rl.config import AppConfig

    raw = _synthetic_raw(rows=800)
    frame = build_feature_frame(raw, FeatureConfig())
    modeling_frame = build_price_model_frame(frame)
    assert not modeling_frame.empty

    cfg = AppConfig()
    cfg.research.min_train_rows = 200
    cfg.research.min_test_rows = 100
    artifacts = train_gradient_boosting_models(frame, cfg)

    assert "spot" in artifacts.metrics
    assert "futures" in artifacts.metrics
    assert artifacts.feature_columns[-2:] == ["spot_close", "futures_close"]


def test_catboost_training_reports_basis_change_metrics() -> None:
    from kucoin_near_basis_rl.catboost_model import train_catboost_models
    from kucoin_near_basis_rl.catboost_classifier_model import train_catboost_classifier
    from kucoin_near_basis_rl.config import AppConfig
    from kucoin_near_basis_rl.ensemble_model import evaluate_best_ensemble, train_best_ensemble

    raw = _synthetic_raw(rows=800)
    frame = build_feature_frame(raw, FeatureConfig())
    cfg = AppConfig()
    cfg.research.min_train_rows = 200
    cfg.research.min_test_rows = 100
    artifacts = train_catboost_models(frame, cfg)

    assert "basis_change" in artifacts.metrics
    classifier_artifacts = train_catboost_classifier(frame, cfg)
    assert "accuracy" in classifier_artifacts.metrics
    ensemble_artifacts = train_best_ensemble(frame, cfg, candidate_weights=(0.0, 0.1))
    ensemble_result = evaluate_best_ensemble(frame, cfg, ensemble_artifacts)
    assert ensemble_result.metrics["strategy"] == "best_ensemble"


def test_forecast_split_avoids_cross_boundary_target_leakage() -> None:
    from kucoin_near_basis_rl.config import AppConfig

    raw = _synthetic_raw(rows=900)
    frame = build_feature_frame(raw, FeatureConfig())
    cfg = AppConfig()
    cfg.research.min_train_rows = 200
    cfg.research.min_test_rows = 100
    split = split_feature_frame(frame, cfg)

    assert split.train_feature_frame.iloc[-1]["timestamp"] < split.test_feature_frame.iloc[0]["timestamp"]
    assert split.train_modeling_frame.iloc[-1]["timestamp"] < split.test_feature_frame.iloc[0]["timestamp"]


def test_qlearning_training_runs() -> None:
    raw = _synthetic_raw()
    frame = build_feature_frame(raw, FeatureConfig())
    baseline = BaselinePolicy(enter_zscore=1.2, exit_zscore=0.25)

    baseline_positions = []
    position = 0
    for row in frame.itertuples():
        position = baseline.decide_position(float(row.basis_zscore), position)
        baseline_positions.append(position)

    env = BasisTradingEnv(
        feature_frame=frame,
        observation_columns=FEATURE_COLUMNS,
        fee_rate_per_rebalance=0.0005,
        risk_penalty=0.0001,
    )
    bins = build_quantile_bins(frame, FEATURE_COLUMNS, [0.1, 0.3, 0.5, 0.7, 0.9])
    discretizer = StateDiscretizer(bin_edges=bins)
    agent = QLearningAgent(num_actions=3, alpha=0.1, gamma=0.98, seed=1)
    rewards = train_qlearning(
        env=env,
        agent=agent,
        discretizer=discretizer,
        baseline_positions=baseline_positions,
        episodes=5,
        epsilon_start=0.25,
        epsilon_end=0.05,
        epsilon_decay=0.9,
        imitation_start=0.25,
        imitation_end=0.05,
        imitation_decay=0.9,
        baseline_bonus=0.00005,
        max_steps_per_episode=300,
    )
    assert len(rewards) == 5
    assert len(agent.q_table) > 0


def test_qlearning_training_runs_without_baseline_guidance() -> None:
    raw = _synthetic_raw()
    frame = build_feature_frame(raw, FeatureConfig())
    env = BasisTradingEnv(
        feature_frame=frame,
        observation_columns=FEATURE_COLUMNS,
        fee_rate_per_rebalance=0.0005,
        risk_penalty=0.0001,
    )
    bins = build_quantile_bins(frame, FEATURE_COLUMNS, [0.1, 0.3, 0.5, 0.7, 0.9])
    discretizer = StateDiscretizer(bin_edges=bins)
    agent = QLearningAgent(num_actions=3, alpha=0.1, gamma=0.98, seed=2)
    rewards = train_qlearning(
        env=env,
        agent=agent,
        discretizer=discretizer,
        baseline_positions=None,
        episodes=3,
        epsilon_start=0.2,
        epsilon_end=0.05,
        epsilon_decay=0.9,
        imitation_start=0.0,
        imitation_end=0.0,
        imitation_decay=1.0,
        baseline_bonus=0.0,
        max_steps_per_episode=300,
    )
    assert len(rewards) == 3
    assert len(agent.q_table) > 0


def test_prepare_frames_adds_meta_signal_columns() -> None:
    from kucoin_near_basis_rl.config import AppConfig
    from kucoin_near_basis_rl.feature_pipeline import prepare_frames

    raw = _synthetic_raw(rows=800)
    cfg = AppConfig()
    cfg.research.min_train_rows = 200
    cfg.research.min_test_rows = 100
    prepared = prepare_frames(raw, cfg)

    for col in ("meta_signal_direction", "meta_signal_strength", "meta_signal_alignment_flag"):
        assert col in prepared.train_frame.columns
        assert col in prepared.test_frame.columns
    assert np.isfinite(prepared.train_frame["meta_signal_strength"]).all()


def test_construct_with_supported_kwargs_handles_sdk_signature_variants() -> None:
    class OldStyleClient:
        def __init__(self, key: str, secret: str, passphrase: str) -> None:
            self.key = key
            self.secret = secret
            self.passphrase = passphrase

    class NewStyleClient:
        def __init__(
            self,
            key: str,
            secret: str,
            passphrase: str,
            is_sandbox: bool = False,
            url: str | None = None,
        ) -> None:
            self.key = key
            self.secret = secret
            self.passphrase = passphrase
            self.is_sandbox = is_sandbox
            self.url = url

    kwargs = {
        "key": "k",
        "secret": "s",
        "passphrase": "p",
        "is_sandbox": True,
        "url": "https://example.com",
        "unexpected": "ignored",
    }
    old = KuCoinExecutionClient._construct_with_supported_kwargs(OldStyleClient, **kwargs)
    new = KuCoinExecutionClient._construct_with_supported_kwargs(NewStyleClient, **kwargs)

    assert old.key == "k"
    assert old.secret == "s"
    assert old.passphrase == "p"
    assert new.is_sandbox is True
    assert new.url == "https://example.com"
