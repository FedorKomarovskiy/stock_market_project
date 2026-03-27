from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig, load_config
from .features import build_feature_frame
from .forecast_strategy import PRICE_MODEL_FEATURE_COLUMNS, regression_metrics, split_feature_frame
from .kucoin_api import KuCoinPublicDataClient


@dataclass
class RegressionStump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        feature = x[:, self.feature_index]
        return np.where(feature <= self.threshold, self.left_value, self.right_value)


@dataclass
class GradientBoostingRegressorLite:
    n_estimators: int = 60
    learning_rate: float = 0.05
    min_samples_leaf: int = 24
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    initial_prediction: float = 0.0
    stumps: list[RegressionStump] = field(default_factory=list)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "GradientBoostingRegressorLite":
        features = x.to_numpy(dtype=np.float64)
        target = y.to_numpy(dtype=np.float64)
        if features.ndim != 2 or len(target) != len(features):
            raise ValueError("Input shapes are invalid for gradient boosting training.")

        self.initial_prediction = float(np.mean(target))
        current_pred = np.full(len(target), self.initial_prediction, dtype=np.float64)
        self.stumps = []

        for _ in range(self.n_estimators):
            residual = target - current_pred
            stump = self._fit_best_stump(features, residual)
            if stump is None:
                break
            current_pred += self.learning_rate * stump.predict(features)
            self.stumps.append(stump)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        features = x.to_numpy(dtype=np.float64)
        pred = np.full(len(features), self.initial_prediction, dtype=np.float64)
        for stump in self.stumps:
            pred += self.learning_rate * stump.predict(features)
        return pred

    def _fit_best_stump(self, x: np.ndarray, residual: np.ndarray) -> RegressionStump | None:
        best_stump: RegressionStump | None = None
        best_loss = float("inf")
        for feature_index in range(x.shape[1]):
            column = x[:, feature_index]
            thresholds = sorted(set(float(v) for v in np.quantile(column, self.quantiles)))
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                left_value = float(np.mean(residual[left_mask]))
                right_value = float(np.mean(residual[right_mask]))
                loss = float(
                    np.sum(np.square(residual[left_mask] - left_value))
                    + np.sum(np.square(residual[right_mask] - right_value))
                )
                if loss < best_loss:
                    best_loss = loss
                    best_stump = RegressionStump(
                        feature_index=feature_index,
                        threshold=float(threshold),
                        left_value=left_value,
                        right_value=right_value,
                    )
        return best_stump


@dataclass
class GBTrainingArtifacts:
    feature_frame: pd.DataFrame
    train_modeling_frame: pd.DataFrame
    test_modeling_frame: pd.DataFrame
    feature_columns: list[str]
    spot_model: GradientBoostingRegressorLite
    futures_model: GradientBoostingRegressorLite
    metrics: dict[str, dict[str, float]]


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def default_three_year_window() -> tuple[datetime, datetime]:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365 * 3)
    return start_dt, end_dt


def train_gradient_boosting_models(
    feature_frame: pd.DataFrame,
    cfg: AppConfig,
) -> GBTrainingArtifacts:
    split = split_feature_frame(feature_frame, cfg)
    feature_columns = list(PRICE_MODEL_FEATURE_COLUMNS)
    train_frame = split.train_modeling_frame
    test_frame = split.test_modeling_frame

    x_train = train_frame[feature_columns]
    x_test = test_frame[feature_columns]

    spot_model = GradientBoostingRegressorLite(n_estimators=80, learning_rate=0.05, min_samples_leaf=24)
    futures_model = GradientBoostingRegressorLite(n_estimators=80, learning_rate=0.05, min_samples_leaf=24)
    spot_model.fit(x_train, train_frame["target_spot_close"])
    futures_model.fit(x_train, train_frame["target_futures_close"])

    metrics = {
        "spot": regression_metrics(test_frame["target_spot_close"], spot_model.predict(x_test)),
        "futures": regression_metrics(test_frame["target_futures_close"], futures_model.predict(x_test)),
    }
    return GBTrainingArtifacts(
        feature_frame=feature_frame,
        train_modeling_frame=train_frame,
        test_modeling_frame=test_frame,
        feature_columns=feature_columns,
        spot_model=spot_model,
        futures_model=futures_model,
        metrics=metrics,
    )


def save_gradient_boosting_artifact(
    path: str | Path,
    artifacts: GBTrainingArtifacts,
    metadata: dict[str, object] | None = None,
) -> None:
    payload = {
        "feature_columns": artifacts.feature_columns,
        "train_rows": int(len(artifacts.train_modeling_frame)),
        "test_rows": int(len(artifacts.test_modeling_frame)),
        "metrics": artifacts.metrics,
        "spot_model": artifacts.spot_model,
        "futures_model": artifacts.futures_model,
        "metadata": metadata or {},
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)


def run_gradient_boosting_pipeline(
    config_path: str,
    model_out: str,
    report_dir: str,
    source_csv: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    raw_out: str | None = None,
    features_out: str | None = None,
) -> dict[str, object]:
    cfg = load_config(config_path)
    if source_csv:
        raw_frame = pd.read_csv(source_csv, parse_dates=["timestamp"])
        if raw_frame["timestamp"].dt.tz is None:
            raw_frame["timestamp"] = raw_frame["timestamp"].dt.tz_localize("UTC")
    else:
        data_client = KuCoinPublicDataClient(cfg.api)
        if start_iso and end_iso:
            start_dt = _parse_dt(start_iso)
            end_dt = _parse_dt(end_iso)
        else:
            start_dt, end_dt = default_three_year_window()
        raw_frame = data_client.fetch_merged_candles(cfg.data, start_dt=start_dt, end_dt=end_dt)

    if raw_out:
        raw_path = Path(raw_out)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_frame.to_csv(raw_path, index=False)

    feature_frame = build_feature_frame(raw_frame, cfg.features)
    artifacts = train_gradient_boosting_models(feature_frame, cfg)

    if features_out:
        features_path = Path(features_out)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [
                artifacts.train_modeling_frame.assign(split="train"),
                artifacts.test_modeling_frame.assign(split="test"),
            ],
            ignore_index=True,
        ).to_csv(features_path, index=False)

    save_gradient_boosting_artifact(
        path=model_out,
        artifacts=artifacts,
        metadata={
            "config_path": str(config_path),
            "rows_used": int(len(artifacts.train_modeling_frame) + len(artifacts.test_modeling_frame)),
            "train_rows": int(len(artifacts.train_modeling_frame)),
            "test_rows": int(len(artifacts.test_modeling_frame)),
            "train_start": str(artifacts.train_modeling_frame.iloc[0]["timestamp"]),
            "train_end": str(artifacts.train_modeling_frame.iloc[-1]["timestamp"]),
            "test_start": str(artifacts.test_modeling_frame.iloc[0]["timestamp"]),
            "test_end": str(artifacts.test_modeling_frame.iloc[-1]["timestamp"]),
        },
    )

    latest_row = artifacts.test_modeling_frame.iloc[-1]
    latest_features = latest_row[artifacts.feature_columns].to_frame().T
    next_spot_prediction = float(artifacts.spot_model.predict(latest_features)[0])
    next_futures_prediction = float(artifacts.futures_model.predict(latest_features)[0])

    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "pair": {
            "spot_symbol": cfg.data.spot_symbol,
            "futures_symbol": cfg.data.futures_symbol,
        },
        "rows": {
            "raw": int(len(raw_frame)),
            "features": int(len(feature_frame)),
            "modeling": int(len(artifacts.train_modeling_frame) + len(artifacts.test_modeling_frame)),
            "train": int(len(artifacts.train_modeling_frame)),
            "test": int(len(artifacts.test_modeling_frame)),
        },
        "model_path": str(Path(model_out).resolve()),
        "feature_columns": artifacts.feature_columns,
        "metrics": artifacts.metrics,
        "latest_prediction": {
            "timestamp": str(latest_row["timestamp"]),
            "spot_close": float(latest_row["spot_close"]),
            "futures_close": float(latest_row["futures_close"]),
            "predicted_next_spot_close": next_spot_prediction,
            "predicted_next_futures_close": next_futures_prediction,
        },
    }
    summary_path = report_root / "gb_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary
