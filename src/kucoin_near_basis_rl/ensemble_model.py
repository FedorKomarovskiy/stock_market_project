from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .catboost_model import CatBoostTrainingArtifacts, train_catboost_models
from .config import AppConfig
from .features import build_feature_frame
from .forecast_strategy import (
    evaluate_forecast_positions,
    predicted_basis_change,
    select_best_threshold,
    split_feature_frame,
)
from .gb_model import GBTrainingArtifacts, train_gradient_boosting_models


@dataclass
class EnsembleArtifacts:
    gb: GBTrainingArtifacts
    catboost: CatBoostTrainingArtifacts
    catboost_weight: float
    train_threshold: float
    train_result: BacktestResult


def train_best_ensemble(
    feature_frame: pd.DataFrame,
    cfg: AppConfig,
    candidate_weights: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> EnsembleArtifacts:
    split = split_feature_frame(feature_frame, cfg)
    gb = train_gradient_boosting_models(feature_frame, cfg)
    catboost = train_catboost_models(feature_frame, cfg)

    train_basis_frame = split.train_feature_frame.iloc[:-1].reset_index(drop=True)
    gb_train_pred = predicted_basis_change(
        gb.train_modeling_frame,
        gb.spot_model.predict(gb.train_modeling_frame[gb.feature_columns]),
        gb.futures_model.predict(gb.train_modeling_frame[gb.feature_columns]),
    )
    cat_train_pred = np.asarray(
        catboost.basis_model.predict(train_basis_frame[catboost.feature_columns]),
        dtype=float,
    )

    best_weight = float(candidate_weights[0])
    combined_pred = (1.0 - best_weight) * gb_train_pred + best_weight * cat_train_pred
    best_threshold, best_result = select_best_threshold(
        feature_frame=train_basis_frame,
        predicted_change=combined_pred,
        cfg=cfg,
        strategy_name="ensemble_train",
    )
    best_key = (
        float(best_result.metrics["sharpe"]),
        float(best_result.metrics["total_return"]),
        float(best_result.metrics["max_drawdown"]),
    )

    for weight in candidate_weights[1:]:
        combined_pred = (1.0 - float(weight)) * gb_train_pred + float(weight) * cat_train_pred
        threshold, result = select_best_threshold(
            feature_frame=train_basis_frame,
            predicted_change=combined_pred,
            cfg=cfg,
            strategy_name="ensemble_train",
        )
        current_key = (
            float(result.metrics["sharpe"]),
            float(result.metrics["total_return"]),
            float(result.metrics["max_drawdown"]),
        )
        if current_key > best_key:
            best_weight = float(weight)
            best_threshold = float(threshold)
            best_result = result
            best_key = current_key

    return EnsembleArtifacts(
        gb=gb,
        catboost=catboost,
        catboost_weight=best_weight,
        train_threshold=best_threshold,
        train_result=best_result,
    )


def evaluate_best_ensemble(
    feature_frame: pd.DataFrame,
    cfg: AppConfig,
    artifacts: EnsembleArtifacts,
) -> BacktestResult:
    split = split_feature_frame(feature_frame, cfg)
    test_basis_frame = split.test_feature_frame.iloc[:-1].reset_index(drop=True)
    gb_test_pred = predicted_basis_change(
        artifacts.gb.test_modeling_frame,
        artifacts.gb.spot_model.predict(artifacts.gb.test_modeling_frame[artifacts.gb.feature_columns]),
        artifacts.gb.futures_model.predict(artifacts.gb.test_modeling_frame[artifacts.gb.feature_columns]),
    )
    cat_test_pred = np.asarray(
        artifacts.catboost.basis_model.predict(test_basis_frame[artifacts.catboost.feature_columns]),
        dtype=float,
    )
    combined_pred = (1.0 - artifacts.catboost_weight) * gb_test_pred + artifacts.catboost_weight * cat_test_pred
    return evaluate_forecast_positions(
        feature_frame=test_basis_frame,
        predicted_change=combined_pred,
        threshold=artifacts.train_threshold,
        cfg=cfg,
        strategy_name="best_ensemble",
    )
