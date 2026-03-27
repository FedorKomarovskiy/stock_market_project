from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
        description="Walk-forward comparison of top GB and enhanced DQN meta-controller strategies."
    )
    parser.add_argument("--config", default="config/project_near_hourly.json")
    parser.add_argument("--source-csv", default="", help="Optional local CSV. Ignored if --start and --end are set.")
    parser.add_argument("--report-dir", default="reports/top5_dqn_walkforward")
    parser.add_argument("--episodes", type=int, default=45, help="Enhanced DQN episodes.")
    parser.add_argument("--start", default="", help="UTC ISO start if downloading from KuCoin.")
    parser.add_argument("--end", default="", help="UTC ISO end if downloading from KuCoin.")
    parser.add_argument("--folds", type=int, default=2, help="Number of expanding walk-forward folds.")
    parser.add_argument("--reward-horizon", type=int, default=6, help="Enhanced DQN reward horizon.")
    return parser.parse_args()


def build_walkforward_slices(
    total_rows: int,
    min_train_rows: int,
    min_test_rows: int,
    folds: int,
) -> list[tuple[int, int]]:
    num_folds = max(1, int(folds))
    test_rows = int(min_test_rows)
    while num_folds > 1 and total_rows - (num_folds * test_rows) < min_train_rows:
        num_folds -= 1
    initial_train = total_rows - (num_folds * test_rows)
    if initial_train < min_train_rows:
        raise RuntimeError("Not enough rows for walk-forward evaluation with current settings.")
    slices: list[tuple[int, int]] = []
    train_end = initial_train
    for _ in range(num_folds):
        test_end = min(total_rows, train_end + test_rows)
        if test_end - train_end < min_test_rows:
            break
        slices.append((train_end, test_end))
        train_end = test_end
    if not slices:
        raise RuntimeError("Unable to build any walk-forward folds.")
    return slices


def main() -> int:
    configure_console_utf8()
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    _ensure_pythonpath(repo_root)

    from kucoin_near_basis_rl.backtest import (
        apply_kelly_overlay,
        backtest_positions,
        combine_backtest_results,
        estimate_kelly_fraction,
    )
    from kucoin_near_basis_rl.config import load_config
    from kucoin_near_basis_rl.enhanced_meta_dqn import (
        rollout_enhanced_meta_dqn_positions,
        train_enhanced_meta_dqn_agent,
    )
    from kucoin_near_basis_rl.feature_pipeline import (
        RL_META_OBSERVATION_COLUMNS,
        add_gb_auxiliary_features,
        build_enriched_feature_frame,
    )
    from kucoin_near_basis_rl.forecast_strategy import positions_from_predicted_change, select_best_threshold
    from kucoin_near_basis_rl.kucoin_api import KuCoinPublicDataClient
    from kucoin_near_basis_rl.research import _parse_dt

    cfg = load_config(str((repo_root / args.config).resolve()))
    source_csv = str(args.source_csv).strip()
    if args.start and args.end:
        data_client = KuCoinPublicDataClient(cfg.api)
        raw_frame = data_client.fetch_merged_candles(
            cfg.data,
            start_dt=_parse_dt(args.start),
            end_dt=_parse_dt(args.end),
        )
    elif source_csv:
        source_path = (repo_root / source_csv).resolve()
        raw_frame = pd.read_csv(source_path, parse_dates=["timestamp"])
        if raw_frame["timestamp"].dt.tz is None:
            raw_frame["timestamp"] = raw_frame["timestamp"].dt.tz_localize("UTC")
    else:
        raise RuntimeError("Either --source-csv or both --start and --end must be provided.")

    feature_frame = build_enriched_feature_frame(raw_frame, cfg)
    slices = build_walkforward_slices(
        total_rows=len(feature_frame),
        min_train_rows=int(cfg.research.min_train_rows),
        min_test_rows=int(cfg.research.min_test_rows),
        folds=int(args.folds),
    )

    kelly_max_fraction = 1.0
    kelly_fraction_multiplier = 0.25
    results_by_strategy: dict[str, list] = {
        "gradient_boosting": [],
        "gradient_boosting_kelly": [],
        "gb_plus_baseline_filter": [],
        "gb_plus_baseline_filter_kelly": [],
        "dqn_enhanced_meta_controller": [],
        "dqn_enhanced_meta_controller_kelly": [],
    }
    fold_summaries: list[dict[str, object]] = []

    for fold_idx, (train_end, test_end) in enumerate(slices):
        train_feature = feature_frame.iloc[:train_end].reset_index(drop=True)
        test_feature = feature_frame.iloc[train_end:test_end].reset_index(drop=True)
        train_prepared, test_prepared, _gb_columns, _gb_threshold = add_gb_auxiliary_features(train_feature, test_feature, cfg)

        gb_threshold, _ = select_best_threshold(
            feature_frame=train_prepared,
            predicted_change=train_prepared["gb_pred_basis_change"].to_numpy(dtype=float),
            cfg=cfg,
            strategy_name=f"gradient_boosting_fold_{fold_idx}",
        )
        gb_train_positions = positions_from_predicted_change(
            predicted_change=train_prepared["gb_pred_basis_change"].to_numpy(dtype=float),
            frame_len=len(train_prepared),
            threshold=gb_threshold,
        )
        gb_test_positions = positions_from_predicted_change(
            predicted_change=test_prepared["gb_pred_basis_change"].to_numpy(dtype=float),
            frame_len=len(test_prepared),
            threshold=gb_threshold,
        )
        gb_result = backtest_positions(
            feature_frame=test_prepared,
            positions=gb_test_positions,
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            initial_capital=cfg.research.initial_capital,
            strategy_name="gradient_boosting",
        )
        gb_kelly_fraction = estimate_kelly_fraction(
            feature_frame=train_prepared,
            positions=gb_train_positions,
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            max_fraction=kelly_max_fraction,
            fraction_multiplier=kelly_fraction_multiplier,
            min_active_rows=36,
        )
        gb_kelly_result = backtest_positions(
            feature_frame=test_prepared,
            positions=apply_kelly_overlay(gb_test_positions, gb_kelly_fraction),
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            initial_capital=cfg.research.initial_capital,
            strategy_name="gradient_boosting_kelly",
        )
        results_by_strategy["gradient_boosting"].append(gb_result)
        results_by_strategy["gradient_boosting_kelly"].append(gb_kelly_result)

        gb_baseline_train_positions: list[float] = []
        gb_baseline_test_positions: list[float] = []
        for row in train_prepared.itertuples():
            if int(row.baseline_position) == int(float(row.gb_pred_direction)) and float(row.gb_high_confidence_flag) > 0.0:
                gb_baseline_train_positions.append(float(row.gb_pred_direction))
            else:
                gb_baseline_train_positions.append(0.0)
        for row in test_prepared.itertuples():
            if int(row.baseline_position) == int(float(row.gb_pred_direction)) and float(row.gb_high_confidence_flag) > 0.0:
                gb_baseline_test_positions.append(float(row.gb_pred_direction))
            else:
                gb_baseline_test_positions.append(0.0)
        gb_baseline_result = backtest_positions(
            feature_frame=test_prepared,
            positions=gb_baseline_test_positions,
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            initial_capital=cfg.research.initial_capital,
            strategy_name="gb_plus_baseline_filter",
        )
        gb_baseline_kelly_fraction = estimate_kelly_fraction(
            feature_frame=train_prepared,
            positions=gb_baseline_train_positions,
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            max_fraction=kelly_max_fraction,
            fraction_multiplier=kelly_fraction_multiplier,
            min_active_rows=36,
        )
        gb_baseline_kelly_result = backtest_positions(
            feature_frame=test_prepared,
            positions=apply_kelly_overlay(gb_baseline_test_positions, gb_baseline_kelly_fraction),
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            initial_capital=cfg.research.initial_capital,
            strategy_name="gb_plus_baseline_filter_kelly",
        )
        results_by_strategy["gb_plus_baseline_filter"].append(gb_baseline_result)
        results_by_strategy["gb_plus_baseline_filter_kelly"].append(gb_baseline_kelly_result)

        dqn_artifacts = train_enhanced_meta_dqn_agent(
            train_frame=train_prepared,
            observation_columns=RL_META_OBSERVATION_COLUMNS,
            episodes=int(args.episodes),
            reward_horizon=int(args.reward_horizon),
        )
        dqn_train_positions = rollout_enhanced_meta_dqn_positions(train_prepared, dqn_artifacts)
        dqn_test_positions = rollout_enhanced_meta_dqn_positions(test_prepared, dqn_artifacts)
        dqn_result = backtest_positions(
            feature_frame=test_prepared,
            positions=dqn_test_positions["target_position"].tolist(),
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            initial_capital=cfg.research.initial_capital,
            strategy_name="dqn_enhanced_meta_controller",
        )
        dqn_kelly_fraction = estimate_kelly_fraction(
            feature_frame=train_prepared,
            positions=dqn_train_positions["target_position"].tolist(),
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            max_fraction=kelly_max_fraction,
            fraction_multiplier=kelly_fraction_multiplier,
            min_active_rows=36,
        )
        dqn_kelly_result = backtest_positions(
            feature_frame=test_prepared,
            positions=apply_kelly_overlay(dqn_test_positions["target_position"].tolist(), dqn_kelly_fraction),
            fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
            risk_penalty=cfg.execution.risk_penalty,
            initial_capital=cfg.research.initial_capital,
            strategy_name="dqn_enhanced_meta_controller_kelly",
        )
        results_by_strategy["dqn_enhanced_meta_controller"].append(dqn_result)
        results_by_strategy["dqn_enhanced_meta_controller_kelly"].append(dqn_kelly_result)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_prepared)),
                "test_rows": int(len(test_prepared)),
                "gb_threshold": float(gb_threshold),
                "gb_kelly_fraction": float(gb_kelly_fraction),
                "gb_baseline_kelly_fraction": float(gb_baseline_kelly_fraction),
                "dqn_kelly_fraction": float(dqn_kelly_fraction),
                "dqn_episode_reward_last_5": float(
                    sum(dqn_artifacts.episode_rewards[-5:]) / max(1, len(dqn_artifacts.episode_rewards[-5:]))
                ),
            }
        )

    combined_results = {
        strategy: combine_backtest_results(
            results=result_list,
            initial_capital=cfg.research.initial_capital,
            strategy_name=strategy,
        )
        for strategy, result_list in results_by_strategy.items()
    }
    comparison_df = pd.DataFrame([result.metrics for result in combined_results.values()])
    comparison_df = comparison_df.sort_values(
        by=["total_return", "sharpe", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top5_df = comparison_df.head(5).copy()

    report_root = (repo_root / args.report_dir).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    full_path = report_root / "comparison_metrics_full.csv"
    top5_path = report_root / "comparison_metrics_top5.csv"
    comparison_df.to_csv(full_path, index=False)
    top5_df.to_csv(top5_path, index=False)

    summary = {
        "rows": {
            "raw": int(len(raw_frame)),
            "features": int(len(feature_frame)),
        },
        "walkforward_slices": [
            {"train_end": int(train_end), "test_end": int(test_end)} for train_end, test_end in slices
        ],
        "fold_summaries": fold_summaries,
        "kelly_fraction_multiplier": kelly_fraction_multiplier,
        "kelly_max_fraction": kelly_max_fraction,
        "comparison_metrics_full_path": str(full_path),
        "comparison_metrics_top5_path": str(top5_path),
        "top5_metrics": top5_df.to_dict(orient="records"),
    }
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
