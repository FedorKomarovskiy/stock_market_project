from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import AppConfig, load_config
from .features import build_feature_frame
from .forecast_strategy import PRICE_MODEL_FEATURE_COLUMNS, regression_metrics, split_feature_frame
from .kucoin_api import KuCoinPublicDataClient


@dataclass
class CatBoostTrainingArtifacts:
    feature_frame: pd.DataFrame
    train_modeling_frame: pd.DataFrame
    test_modeling_frame: pd.DataFrame
    feature_columns: list[str]
    spot_model: CatBoostRegressor
    futures_model: CatBoostRegressor
    basis_model: CatBoostRegressor
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


def train_catboost_models(feature_frame: pd.DataFrame, cfg: AppConfig) -> CatBoostTrainingArtifacts:
    split = split_feature_frame(feature_frame, cfg)
    feature_columns = list(PRICE_MODEL_FEATURE_COLUMNS)
    train_frame = split.train_modeling_frame
    test_frame = split.test_modeling_frame
    train_basis_frame = split.train_feature_frame.copy()
    test_basis_frame = split.test_feature_frame.copy()
    train_basis_frame["target_basis_change"] = train_basis_frame["basis"].shift(-1) - train_basis_frame["basis"]
    test_basis_frame["target_basis_change"] = test_basis_frame["basis"].shift(-1) - test_basis_frame["basis"]
    train_basis_frame = train_basis_frame.dropna().reset_index(drop=True)
    test_basis_frame = test_basis_frame.dropna().reset_index(drop=True)

    x_train = train_frame[feature_columns]
    x_test = test_frame[feature_columns]
    x_train_basis = train_basis_frame[feature_columns]
    x_test_basis = test_basis_frame[feature_columns]

    spot_model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    futures_model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    basis_model = CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    spot_model.fit(x_train, train_frame["target_spot_close"])
    futures_model.fit(x_train, train_frame["target_futures_close"])
    basis_model.fit(x_train_basis, train_basis_frame["target_basis_change"])

    return CatBoostTrainingArtifacts(
        feature_frame=feature_frame,
        train_modeling_frame=train_frame,
        test_modeling_frame=test_frame,
        feature_columns=feature_columns,
        spot_model=spot_model,
        futures_model=futures_model,
        basis_model=basis_model,
        metrics={
            "spot": regression_metrics(test_frame["target_spot_close"], spot_model.predict(x_test)),
            "futures": regression_metrics(test_frame["target_futures_close"], futures_model.predict(x_test)),
            "basis_change": regression_metrics(
                test_basis_frame["target_basis_change"],
                np.asarray(basis_model.predict(x_test_basis), dtype=float),
            ),
        },
    )


def run_catboost_pipeline(
    config_path: str,
    model_dir: str,
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
    artifacts = train_catboost_models(feature_frame, cfg)

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spot_model_path = out_dir / "catboost_spot.cbm"
    futures_model_path = out_dir / "catboost_futures.cbm"
    basis_model_path = out_dir / "catboost_basis_change.cbm"
    artifacts.spot_model.save_model(str(spot_model_path))
    artifacts.futures_model.save_model(str(futures_model_path))
    artifacts.basis_model.save_model(str(basis_model_path))

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

    latest_row = artifacts.test_modeling_frame.iloc[-1]
    latest_features = latest_row[artifacts.feature_columns].to_frame().T
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
        "model_dir": str(out_dir.resolve()),
        "feature_columns": artifacts.feature_columns,
        "metrics": artifacts.metrics,
        "latest_prediction": {
            "timestamp": str(latest_row["timestamp"]),
            "spot_close": float(latest_row["spot_close"]),
            "futures_close": float(latest_row["futures_close"]),
            "predicted_next_spot_close": float(artifacts.spot_model.predict(latest_features)[0]),
            "predicted_next_futures_close": float(artifacts.futures_model.predict(latest_features)[0]),
        },
    }
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "catboost_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    summary["spot_model_path"] = str(spot_model_path.resolve())
    summary["futures_model_path"] = str(futures_model_path.resolve())
    summary["basis_model_path"] = str(basis_model_path.resolve())
    return summary
