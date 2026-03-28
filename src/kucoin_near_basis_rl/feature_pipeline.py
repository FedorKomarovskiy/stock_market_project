from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import BaselinePolicy
from .config import AppConfig
from .features import FEATURE_COLUMNS, build_feature_frame
from .finnhub_news import build_news_feature_frame
from .forecast_strategy import split_feature_frame
from .gb_model import GradientBoostingRegressorLite


BASELINE_FEATURE_COLUMNS = [
    "baseline_position",
    "baseline_enter_long_flag",
    "baseline_enter_short_flag",
    "baseline_flat_flag",
    "distance_to_baseline_entry",
    "distance_to_baseline_exit",
]

GB_AUX_COLUMNS = [
    "gb_pred_basis_change",
    "gb_pred_abs_basis_change",
    "gb_pred_direction",
    "gb_high_confidence_flag",
    "gb_confidence_score",
    "rl_regime_active_flag",
    "meta_signal_direction",
    "meta_signal_strength",
    "meta_signal_alignment_flag",
]

RL_OBSERVATION_COLUMNS = list(FEATURE_COLUMNS) + BASELINE_FEATURE_COLUMNS + GB_AUX_COLUMNS
RL_COMPACT_OBSERVATION_COLUMNS = [
    "basis",
    "basis_return",
    "basis_zscore",
    "basis_volatility",
    "basis_momentum",
    "basis_ema_gap",
    "volume_imbalance",
    "rolling_correlation",
    "news_sentiment_mean_6h",
    "news_sentiment_sum_24h",
    "news_count_24h",
    "news_cryptopanic_count_24h",
    "baseline_position",
    "baseline_enter_long_flag",
    "baseline_enter_short_flag",
    "distance_to_baseline_entry",
    "gb_pred_basis_change",
    "gb_pred_abs_basis_change",
    "gb_pred_direction",
    "gb_high_confidence_flag",
    "gb_confidence_score",
    "rl_regime_active_flag",
]

RL_META_OBSERVATION_COLUMNS = [
    "basis",
    "basis_return",
    "basis_zscore",
    "basis_volatility",
    "basis_momentum",
    "volume_imbalance",
    "rolling_correlation",
    "news_sentiment_mean_6h",
    "news_sentiment_sum_24h",
    "news_count_24h",
    "news_cryptopanic_count_24h",
    "baseline_position",
    "baseline_enter_long_flag",
    "baseline_enter_short_flag",
    "distance_to_baseline_entry",
    "distance_to_baseline_exit",
    "gb_pred_basis_change",
    "gb_pred_abs_basis_change",
    "gb_pred_direction",
    "gb_high_confidence_flag",
    "gb_confidence_score",
    "meta_signal_direction",
    "meta_signal_strength",
    "meta_signal_alignment_flag",
    "rl_regime_active_flag",
]


@dataclass
class PreparedFrames:
    feature_frame: pd.DataFrame
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    gb_feature_columns: list[str]
    gb_threshold: float


def build_enriched_feature_frame(raw_frame: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    news_feature_frame = build_news_feature_frame(raw_frame, cfg.news)
    feature_frame = build_feature_frame(raw_frame, cfg.features, news_feature_frame=news_feature_frame)
    return add_baseline_features(feature_frame, cfg)


def add_baseline_features(feature_frame: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    frame = feature_frame.copy()
    policy = BaselinePolicy(
        enter_zscore=cfg.baseline.enter_zscore,
        exit_zscore=cfg.baseline.exit_zscore,
    )
    positions: list[int] = []
    current_position = 0
    for row in frame.itertuples():
        current_position = policy.decide_position(float(row.basis_zscore), current_position)
        positions.append(current_position)

    abs_z = frame["basis_zscore"].abs().astype(float)
    frame["baseline_position"] = positions
    frame["baseline_enter_long_flag"] = (frame["basis_zscore"] <= -float(cfg.baseline.enter_zscore)).astype(float)
    frame["baseline_enter_short_flag"] = (frame["basis_zscore"] >= float(cfg.baseline.enter_zscore)).astype(float)
    frame["baseline_flat_flag"] = (abs_z <= float(cfg.baseline.exit_zscore)).astype(float)
    frame["distance_to_baseline_entry"] = abs_z - float(cfg.baseline.enter_zscore)
    frame["distance_to_baseline_exit"] = float(cfg.baseline.exit_zscore) - abs_z
    return frame


def get_gb_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    return list(FEATURE_COLUMNS) + BASELINE_FEATURE_COLUMNS + ["spot_close", "futures_close"]


def add_gb_auxiliary_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    cfg: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    train = train_frame.copy()
    test = test_frame.copy()
    feature_columns = get_gb_feature_columns(train)

    train_modeling = train.copy()
    test_modeling = test.copy()
    train_modeling["target_spot_close"] = train_modeling["spot_close"].shift(-1)
    train_modeling["target_futures_close"] = train_modeling["futures_close"].shift(-1)
    test_modeling["target_spot_close"] = test_modeling["spot_close"].shift(-1)
    test_modeling["target_futures_close"] = test_modeling["futures_close"].shift(-1)
    train_modeling = train_modeling.dropna().reset_index(drop=True)
    test_modeling = test_modeling.dropna().reset_index(drop=True)

    spot_model = GradientBoostingRegressorLite(n_estimators=80, learning_rate=0.05, min_samples_leaf=24)
    futures_model = GradientBoostingRegressorLite(n_estimators=80, learning_rate=0.05, min_samples_leaf=24)
    spot_model.fit(train_modeling[feature_columns], train_modeling["target_spot_close"])
    futures_model.fit(train_modeling[feature_columns], train_modeling["target_futures_close"])

    train_pred_change = _predicted_basis_change(
        frame=train_modeling,
        predicted_spot=spot_model.predict(train_modeling[feature_columns]),
        predicted_futures=futures_model.predict(train_modeling[feature_columns]),
    )
    test_pred_change = _predicted_basis_change(
        frame=test_modeling,
        predicted_spot=spot_model.predict(test_modeling[feature_columns]),
        predicted_futures=futures_model.predict(test_modeling[feature_columns]),
    )
    threshold = float(np.quantile(np.abs(train_pred_change), 0.67)) if len(train_pred_change) else 0.0

    train = _merge_gb_predictions(train, train_pred_change, threshold)
    test = _merge_gb_predictions(test, test_pred_change, threshold)
    return train, test, feature_columns, threshold


def prepare_frames(raw_frame: pd.DataFrame, cfg: AppConfig) -> PreparedFrames:
    feature_frame = build_enriched_feature_frame(raw_frame, cfg)
    split = split_feature_frame(feature_frame, cfg)
    train_frame, test_frame, gb_feature_columns, gb_threshold = add_gb_auxiliary_features(
        split.train_feature_frame,
        split.test_feature_frame,
        cfg,
    )
    feature_frame_full = pd.concat([train_frame, test_frame], ignore_index=True)
    return PreparedFrames(
        feature_frame=feature_frame_full,
        train_frame=train_frame,
        test_frame=test_frame,
        gb_feature_columns=gb_feature_columns,
        gb_threshold=gb_threshold,
    )


def _predicted_basis_change(
    frame: pd.DataFrame,
    predicted_spot: np.ndarray,
    predicted_futures: np.ndarray,
) -> np.ndarray:
    pred_spot = np.maximum(np.asarray(predicted_spot, dtype=np.float64), 1e-9)
    pred_futures = np.asarray(predicted_futures, dtype=np.float64)
    predicted_basis = (pred_futures - pred_spot) / pred_spot
    current_basis = frame["basis"].to_numpy(dtype=np.float64)
    return predicted_basis - current_basis


def _merge_gb_predictions(frame: pd.DataFrame, predicted_change: np.ndarray, threshold: float) -> pd.DataFrame:
    enriched = frame.copy()
    base = np.zeros(len(enriched), dtype=np.float64)
    base[: len(predicted_change)] = predicted_change
    abs_change = np.abs(base)
    enriched["gb_pred_basis_change"] = base
    enriched["gb_pred_abs_basis_change"] = abs_change
    enriched["gb_pred_direction"] = np.sign(base)
    enriched["gb_high_confidence_flag"] = (abs_change >= float(threshold)).astype(float)
    enriched["gb_confidence_score"] = np.where(threshold > 0.0, abs_change / threshold, abs_change)
    baseline_extreme = (
        (enriched["baseline_enter_long_flag"].astype(float) > 0.0)
        | (enriched["baseline_enter_short_flag"].astype(float) > 0.0)
    ).astype(float)
    enriched["rl_regime_active_flag"] = np.maximum(enriched["gb_high_confidence_flag"].astype(float), baseline_extreme)
    baseline_direction = np.sign(enriched["baseline_position"].to_numpy(dtype=np.float64))
    gb_direction = enriched["gb_pred_direction"].to_numpy(dtype=np.float64)
    gb_confidence = enriched["gb_confidence_score"].to_numpy(dtype=np.float64)
    baseline_strength = np.clip(
        np.maximum(enriched["distance_to_baseline_entry"].to_numpy(dtype=np.float64), 0.0) + 1.0,
        0.0,
        None,
    )
    baseline_active = (baseline_direction != 0.0).astype(np.float64)

    meta_direction = np.where(
        (baseline_active > 0.0) & (gb_direction != 0.0) & (baseline_direction == gb_direction),
        gb_direction,
        np.where(gb_confidence >= 1.0, gb_direction, baseline_direction),
    )
    meta_alignment = (
        (baseline_active > 0.0)
        & (gb_direction != 0.0)
        & (baseline_direction == gb_direction)
    ).astype(np.float64)
    meta_strength = np.where(
        meta_alignment > 0.0,
        gb_confidence + baseline_strength,
        np.where(gb_direction != 0.0, gb_confidence, baseline_strength * baseline_active),
    )
    enriched["meta_signal_direction"] = meta_direction
    enriched["meta_signal_strength"] = meta_strength
    enriched["meta_signal_alignment_flag"] = meta_alignment
    return enriched
