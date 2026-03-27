from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _ensure_pythonpath(repo_root: Path) -> None:
    src_path = str((repo_root / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RL, gradient boosting and CatBoost on the same historical split."
    )
    parser.add_argument("--config", default="config/project_near_hourly.json", help="Path to JSON config.")
    parser.add_argument(
        "--env-file",
        default=".runtime/project.env",
        help="Runtime credentials file. Defaults to .runtime/project.env with .runtime/kucoin.env fallback.",
    )
    parser.add_argument(
        "--source-csv",
        default="data/project_near_hourly_raw.csv",
        help="Local merged OHLCV CSV. If missing, data will be downloaded.",
    )
    parser.add_argument("--report-dir", default="reports/model_comparison", help="Directory for outputs.")
    parser.add_argument(
        "--rl-model-out",
        default="models/project_near_hourly_qlearning_compare.json",
        help="Path to save RL model artifact for this comparison run.",
    )
    parser.add_argument("--start", default="", help="UTC ISO start for download if source CSV is absent.")
    parser.add_argument("--end", default="", help="UTC ISO end for download if source CSV is absent.")
    parser.add_argument("--episodes", type=int, default=0, help="Optional RL episodes override.")
    return parser.parse_args()


def main() -> int:
    configure_console_utf8()
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    _ensure_pythonpath(repo_root)

    from kucoin_near_basis_rl.runtime_env import load_repo_env

    load_repo_env(repo_root, args.env_file, overwrite=False)
    from kucoin_near_basis_rl.backtest import backtest_positions, rollout_rl_positions
    from kucoin_near_basis_rl.catboost_classifier_model import train_catboost_classifier
    from kucoin_near_basis_rl.catboost_model import train_catboost_models
    from kucoin_near_basis_rl.config import load_config
    from kucoin_near_basis_rl.ensemble_model import evaluate_best_ensemble, train_best_ensemble
    from kucoin_near_basis_rl.features import build_feature_frame
    from kucoin_near_basis_rl.forecast_strategy import (
        evaluate_forecast_positions,
        predicted_basis_change,
        select_best_threshold,
        split_feature_frame,
    )
    from kucoin_near_basis_rl.gb_model import train_gradient_boosting_models
    from kucoin_near_basis_rl.kucoin_api import KuCoinPublicDataClient
    from kucoin_near_basis_rl.qlearning import save_model_artifact
    from kucoin_near_basis_rl.research import _parse_dt
    from kucoin_near_basis_rl.train import train_agent_from_features

    cfg = load_config(str((repo_root / args.config).resolve()))
    if args.episodes and args.episodes > 0:
        cfg.rl.episodes = int(args.episodes)

    source_csv = str((repo_root / args.source_csv).resolve()) if args.source_csv else ""
    source_path = Path(source_csv) if source_csv else None
    if source_path and source_path.exists():
        raw_frame = pd.read_csv(source_path, parse_dates=["timestamp"])
        if raw_frame["timestamp"].dt.tz is None:
            raw_frame["timestamp"] = raw_frame["timestamp"].dt.tz_localize("UTC")
    else:
        data_client = KuCoinPublicDataClient(cfg.api)
        if args.start and args.end:
            start_dt = _parse_dt(args.start)
            end_dt = _parse_dt(args.end)
        else:
            start_dt, end_dt = data_client.utc_lookback(cfg.data.lookback_minutes)
        raw_frame = data_client.fetch_merged_candles(cfg.data, start_dt=start_dt, end_dt=end_dt)

    feature_frame = build_feature_frame(raw_frame, cfg.features)
    split = split_feature_frame(feature_frame, cfg)

    rl_artifacts = train_agent_from_features(split.train_feature_frame, cfg)
    save_model_artifact(
        path=str((repo_root / args.rl_model_out).resolve()),
        agent=rl_artifacts.agent,
        discretizer=rl_artifacts.discretizer,
        observation_columns=rl_artifacts.observation_columns,
        metadata={"config_path": args.config},
    )
    rl_positions = rollout_rl_positions(
        feature_frame=split.test_feature_frame,
        agent=rl_artifacts.agent,
        discretizer=rl_artifacts.discretizer,
        observation_columns=rl_artifacts.observation_columns,
    )
    rl_result = backtest_positions(
        feature_frame=split.test_feature_frame,
        positions=rl_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="rl_agent",
    )

    gb_artifacts = train_gradient_boosting_models(feature_frame, cfg)
    gb_train_pred_spot = gb_artifacts.spot_model.predict(gb_artifacts.train_modeling_frame[gb_artifacts.feature_columns])
    gb_train_pred_futures = gb_artifacts.futures_model.predict(gb_artifacts.train_modeling_frame[gb_artifacts.feature_columns])
    gb_train_pred_change = predicted_basis_change(
            gb_artifacts.train_modeling_frame,
            gb_train_pred_spot,
            gb_train_pred_futures,
        )
    gb_threshold, _ = select_best_threshold(
        feature_frame=split.train_feature_frame,
        predicted_change=gb_train_pred_change,
        cfg=cfg,
        strategy_name="gradient_boosting_train",
    )
    gb_result = evaluate_forecast_positions(
        feature_frame=split.test_feature_frame,
        predicted_change=predicted_basis_change(
            gb_artifacts.test_modeling_frame,
            gb_artifacts.spot_model.predict(gb_artifacts.test_modeling_frame[gb_artifacts.feature_columns]),
            gb_artifacts.futures_model.predict(gb_artifacts.test_modeling_frame[gb_artifacts.feature_columns]),
        ),
        threshold=gb_threshold,
        cfg=cfg,
        strategy_name="gradient_boosting",
    )

    cat_artifacts = train_catboost_models(feature_frame, cfg)
    train_basis_frame = split.train_feature_frame.iloc[:-1].reset_index(drop=True)
    test_basis_frame = split.test_feature_frame.iloc[:-1].reset_index(drop=True)
    cat_train_pred_change = np.asarray(
        cat_artifacts.basis_model.predict(train_basis_frame[cat_artifacts.feature_columns]),
        dtype=float,
    )
    cat_threshold, _ = select_best_threshold(
        feature_frame=train_basis_frame,
        predicted_change=cat_train_pred_change,
        cfg=cfg,
        strategy_name="catboost_train",
    )
    cat_result = evaluate_forecast_positions(
        feature_frame=test_basis_frame,
        predicted_change=np.asarray(
            cat_artifacts.basis_model.predict(test_basis_frame[cat_artifacts.feature_columns]),
            dtype=float,
        ),
        threshold=cat_threshold,
        cfg=cfg,
        strategy_name="catboost",
    )

    cat_classifier_artifacts = train_catboost_classifier(feature_frame, cfg)
    classifier_frame = split.test_feature_frame.iloc[:-1].reset_index(drop=True)
    classifier_pred = np.asarray(
        cat_classifier_artifacts.classifier.predict(classifier_frame[cat_classifier_artifacts.feature_columns]),
        dtype=int,
    ).reshape(-1)
    classifier_positions = [{0: -1, 1: 0, 2: 1}[int(value)] for value in classifier_pred]
    classifier_positions.append(0)
    cat_classifier_result = backtest_positions(
        feature_frame=split.test_feature_frame,
        positions=classifier_positions[: len(split.test_feature_frame)],
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="catboost_classifier",
    )

    ensemble_artifacts = train_best_ensemble(feature_frame, cfg)
    ensemble_result = evaluate_best_ensemble(feature_frame, cfg, ensemble_artifacts)

    report_root = (repo_root / args.report_dir).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(
        [rl_result.metrics, gb_result.metrics, cat_result.metrics, cat_classifier_result.metrics, ensemble_result.metrics]
    )
    comparison_path = report_root / "comparison_metrics.csv"
    comparison_df.to_csv(comparison_path, index=False)

    forecast_df = pd.DataFrame(
        [
            {"model": "gradient_boosting_spot", **gb_artifacts.metrics["spot"]},
            {"model": "gradient_boosting_futures", **gb_artifacts.metrics["futures"]},
            {"model": "catboost_spot", **cat_artifacts.metrics["spot"]},
            {"model": "catboost_futures", **cat_artifacts.metrics["futures"]},
            {"model": "catboost_basis_change", **cat_artifacts.metrics["basis_change"]},
            {"model": "catboost_classifier", **cat_classifier_artifacts.metrics},
        ]
    )
    forecast_path = report_root / "forecast_metrics.csv"
    forecast_df.to_csv(forecast_path, index=False)

    summary = {
        "rows": {
            "raw": int(len(raw_frame)),
            "features": int(len(feature_frame)),
            "train": int(len(split.train_feature_frame)),
            "test": int(len(split.test_feature_frame)),
        },
        "comparison_metrics_path": str(comparison_path),
        "forecast_metrics_path": str(forecast_path),
        "trade_metrics": comparison_df.to_dict(orient="records"),
        "forecast_metrics": forecast_df.to_dict(orient="records"),
        "thresholds": {
            "gradient_boosting": float(gb_threshold),
            "catboost": float(cat_threshold),
            "best_ensemble": float(ensemble_artifacts.train_threshold),
        },
        "ensemble": {
            "catboost_weight": float(ensemble_artifacts.catboost_weight),
        },
    }
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
