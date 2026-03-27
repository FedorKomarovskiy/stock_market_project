from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest_positions
from .config import AppConfig
from .features import FEATURE_COLUMNS


PRICE_MODEL_FEATURE_COLUMNS = list(FEATURE_COLUMNS) + ["spot_close", "futures_close"]


@dataclass
class SplitFrames:
    train_feature_frame: pd.DataFrame
    test_feature_frame: pd.DataFrame
    train_modeling_frame: pd.DataFrame
    test_modeling_frame: pd.DataFrame
    split_index: int


def build_price_model_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    frame = feature_frame.copy()
    frame["target_spot_close"] = frame["spot_close"].shift(-1)
    frame["target_futures_close"] = frame["futures_close"].shift(-1)
    return frame.dropna().reset_index(drop=True)


def split_feature_frame(feature_frame: pd.DataFrame, cfg: AppConfig) -> SplitFrames:
    train_fraction = float(cfg.research.train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("research.train_fraction must be between 0 and 1.")
    split_index = int(len(feature_frame) * train_fraction)
    split_index = max(split_index, int(cfg.research.min_train_rows))
    split_index = min(split_index, len(feature_frame) - int(cfg.research.min_test_rows))
    if split_index <= 0 or split_index >= len(feature_frame):
        raise RuntimeError("Unable to create train/test split with current research settings.")

    train_feature_frame = feature_frame.iloc[:split_index].reset_index(drop=True)
    test_feature_frame = feature_frame.iloc[split_index:].reset_index(drop=True)
    train_modeling_frame = build_price_model_frame(train_feature_frame)
    test_modeling_frame = build_price_model_frame(test_feature_frame)
    if train_modeling_frame.empty or test_modeling_frame.empty:
        raise RuntimeError("Not enough rows after target shifting for forecast models.")

    return SplitFrames(
        train_feature_frame=train_feature_frame,
        test_feature_frame=test_feature_frame,
        train_modeling_frame=train_modeling_frame,
        test_modeling_frame=test_modeling_frame,
        split_index=split_index,
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    actual = y_true.to_numpy(dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    residual = actual - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denom = np.maximum(np.abs(actual), 1e-9)
    mape = float(np.mean(np.abs(residual) / denom))
    ss_res = float(np.sum(np.square(residual)))
    centered = actual - float(actual.mean())
    ss_tot = float(np.sum(np.square(centered)))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def predicted_basis_change(
    current_frame: pd.DataFrame,
    predicted_spot_close: np.ndarray,
    predicted_futures_close: np.ndarray,
) -> np.ndarray:
    pred_spot = np.maximum(np.asarray(predicted_spot_close, dtype=np.float64), 1e-9)
    pred_futures = np.asarray(predicted_futures_close, dtype=np.float64)
    predicted_basis = (pred_futures - pred_spot) / pred_spot
    current_basis = current_frame["basis"].to_numpy(dtype=np.float64)
    return predicted_basis - current_basis


def confidence_threshold(predicted_change_train: np.ndarray, quantile: float = 0.67) -> float:
    absolute = np.abs(np.asarray(predicted_change_train, dtype=np.float64))
    if len(absolute) == 0:
        return 0.0
    return float(np.quantile(absolute, quantile))


def positions_from_predicted_change(
    predicted_change: np.ndarray,
    frame_len: int,
    threshold: float,
) -> list[int]:
    positions: list[int] = []
    for value in np.asarray(predicted_change, dtype=np.float64):
        if value > threshold:
            positions.append(1)
        elif value < -threshold:
            positions.append(-1)
        else:
            positions.append(0)
    while len(positions) < frame_len:
        positions.append(0)
    return positions[:frame_len]


def evaluate_forecast_positions(
    feature_frame: pd.DataFrame,
    predicted_change: np.ndarray,
    threshold: float,
    cfg: AppConfig,
    strategy_name: str,
) -> BacktestResult:
    positions = positions_from_predicted_change(
        predicted_change=predicted_change,
        frame_len=len(feature_frame),
        threshold=threshold,
    )
    return backtest_positions(
        feature_frame=feature_frame,
        positions=positions,
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name=strategy_name,
    )


def select_best_threshold(
    feature_frame: pd.DataFrame,
    predicted_change: np.ndarray,
    cfg: AppConfig,
    strategy_name: str,
    quantiles: tuple[float, ...] = (0.5, 0.6, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95),
) -> tuple[float, BacktestResult]:
    absolute = np.abs(np.asarray(predicted_change, dtype=np.float64))
    if len(absolute) == 0:
        result = evaluate_forecast_positions(feature_frame, predicted_change, 0.0, cfg, strategy_name)
        return 0.0, result

    thresholds = sorted(set(float(np.quantile(absolute, q)) for q in quantiles))
    best_threshold = thresholds[0]
    best_result = evaluate_forecast_positions(feature_frame, predicted_change, best_threshold, cfg, strategy_name)
    best_key = (
        float(best_result.metrics["sharpe"]),
        float(best_result.metrics["total_return"]),
        float(best_result.metrics["max_drawdown"]),
    )
    for threshold in thresholds[1:]:
        result = evaluate_forecast_positions(feature_frame, predicted_change, threshold, cfg, strategy_name)
        current_key = (
            float(result.metrics["sharpe"]),
            float(result.metrics["total_return"]),
            float(result.metrics["max_drawdown"]),
        )
        if current_key > best_key:
            best_threshold = threshold
            best_result = result
            best_key = current_key
    return best_threshold, best_result
