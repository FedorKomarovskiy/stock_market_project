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
        description="Compare baseline, GB, RL-first, and RL meta-controller strategies."
    )
    parser.add_argument("--config", default="config/project_near_hourly.json")
    parser.add_argument("--source-csv", default="data/project_near_hourly_raw.csv")
    parser.add_argument("--report-dir", default="reports/rl_first_comparison")
    parser.add_argument("--episodes", type=int, default=45, help="DQN episodes.")
    parser.add_argument("--ppo-timesteps", type=int, default=20000, help="PPO total timesteps.")
    parser.add_argument("--start", default="", help="UTC ISO start if downloading from KuCoin.")
    parser.add_argument("--end", default="", help="UTC ISO end if downloading from KuCoin.")
    return parser.parse_args()


def main() -> int:
    configure_console_utf8()
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    _ensure_pythonpath(repo_root)

    from kucoin_near_basis_rl.backtest import (
        apply_kelly_overlay,
        backtest_positions,
        build_baseline_positions,
        estimate_kelly_fraction,
    )
    from kucoin_near_basis_rl.config import load_config
    from kucoin_near_basis_rl.dqn_agent import rollout_dqn_positions, train_dqn_agent
    from kucoin_near_basis_rl.feature_pipeline import (
        RL_COMPACT_OBSERVATION_COLUMNS,
        RL_META_OBSERVATION_COLUMNS,
        prepare_frames,
    )
    from kucoin_near_basis_rl.meta_controller_agent import (
        rollout_binary_meta_dqn_positions,
        rollout_binary_meta_ppo_positions,
        rollout_meta_dqn_positions,
        rollout_meta_ppo_positions,
        train_binary_meta_dqn_agent,
        train_binary_meta_ppo_agent,
        train_meta_dqn_agent,
        train_meta_ppo_agent,
    )
    from kucoin_near_basis_rl.ppo_agent import rollout_ppo_positions, train_ppo_agent
    from kucoin_near_basis_rl.forecast_strategy import (
        evaluate_forecast_positions,
        positions_from_predicted_change,
        select_best_threshold,
    )
    from kucoin_near_basis_rl.kucoin_api import KuCoinPublicDataClient
    from kucoin_near_basis_rl.research import _parse_dt
    from kucoin_near_basis_rl.train import train_agent_from_features

    cfg = load_config(str((repo_root / args.config).resolve()))
    source_csv = str(args.source_csv).strip()
    source_path = (repo_root / source_csv).resolve() if source_csv else None
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

    prepared = prepare_frames(raw_frame, cfg)

    kelly_max_fraction = float(max(1.0, cfg.execution.leverage))
    kelly_fraction_multiplier = 0.5

    comparison_rows: list[dict[str, float | str]] = []
    kelly_fractions: dict[str, float] = {}

    baseline_train_positions = build_baseline_positions(
        prepared.train_frame,
        enter_zscore=cfg.baseline.enter_zscore,
        exit_zscore=cfg.baseline.exit_zscore,
    )
    baseline_positions = build_baseline_positions(
        prepared.test_frame,
        enter_zscore=cfg.baseline.enter_zscore,
        exit_zscore=cfg.baseline.exit_zscore,
    )
    baseline_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=baseline_positions.tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="baseline_zscore",
    )
    baseline_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=baseline_train_positions.tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["baseline_zscore"] = baseline_kelly
    baseline_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(baseline_positions.tolist(), baseline_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="baseline_zscore_kelly",
    )
    comparison_rows.extend([baseline_result.metrics, baseline_kelly_result.metrics])

    gb_train_pred = prepared.train_frame["gb_pred_basis_change"].to_numpy(dtype=float)
    gb_threshold, _ = select_best_threshold(
        feature_frame=prepared.train_frame,
        predicted_change=gb_train_pred,
        cfg=cfg,
        strategy_name="gradient_boosting_train",
    )
    gb_result = evaluate_forecast_positions(
        feature_frame=prepared.test_frame,
        predicted_change=prepared.test_frame["gb_pred_basis_change"].to_numpy(dtype=float),
        threshold=gb_threshold,
        cfg=cfg,
        strategy_name="gradient_boosting",
    )
    gb_train_positions = positions_from_predicted_change(
        predicted_change=gb_train_pred,
        frame_len=len(prepared.train_frame),
        threshold=gb_threshold,
    )
    gb_test_positions = positions_from_predicted_change(
        predicted_change=prepared.test_frame["gb_pred_basis_change"].to_numpy(dtype=float),
        frame_len=len(prepared.test_frame),
        threshold=gb_threshold,
    )
    gb_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=gb_train_positions,
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["gradient_boosting"] = gb_kelly
    gb_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(gb_test_positions, gb_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="gradient_boosting_kelly",
    )
    comparison_rows.extend([gb_result.metrics, gb_kelly_result.metrics])

    gb_baseline_train_positions: list[int] = []
    gb_baseline_positions: list[int] = []
    for row in prepared.train_frame.itertuples():
        gb_direction = int(float(row.gb_pred_direction))
        if abs(float(row.gb_pred_basis_change)) < gb_threshold:
            gb_baseline_train_positions.append(0)
        elif int(row.baseline_position) == gb_direction:
            gb_baseline_train_positions.append(gb_direction)
        else:
            gb_baseline_train_positions.append(0)
    for row in prepared.test_frame.itertuples():
        gb_direction = int(float(row.gb_pred_direction))
        if abs(float(row.gb_pred_basis_change)) < gb_threshold:
            gb_baseline_positions.append(0)
        elif int(row.baseline_position) == gb_direction:
            gb_baseline_positions.append(gb_direction)
        else:
            gb_baseline_positions.append(0)
    gb_baseline_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=gb_baseline_positions,
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="gb_plus_baseline_filter",
    )
    gb_baseline_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=gb_baseline_train_positions,
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["gb_plus_baseline_filter"] = gb_baseline_kelly
    gb_baseline_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(gb_baseline_positions, gb_baseline_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="gb_plus_baseline_filter_kelly",
    )
    comparison_rows.extend([gb_baseline_result.metrics, gb_baseline_kelly_result.metrics])

    tabular_artifacts = train_agent_from_features(prepared.train_frame, cfg)
    from kucoin_near_basis_rl.backtest import rollout_rl_positions
    tabular_train_positions = rollout_rl_positions(
        feature_frame=prepared.train_frame,
        agent=tabular_artifacts.agent,
        discretizer=tabular_artifacts.discretizer,
        observation_columns=tabular_artifacts.observation_columns,
    )
    tabular_positions = rollout_rl_positions(
        feature_frame=prepared.test_frame,
        agent=tabular_artifacts.agent,
        discretizer=tabular_artifacts.discretizer,
        observation_columns=tabular_artifacts.observation_columns,
    )
    tabular_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=tabular_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="tabular_rl",
    )
    tabular_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=tabular_train_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["tabular_rl"] = tabular_kelly
    tabular_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(tabular_positions["target_position"].tolist(), tabular_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="tabular_rl_kelly",
    )
    comparison_rows.extend([tabular_result.metrics, tabular_kelly_result.metrics])

    dqn_artifacts = train_dqn_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_COMPACT_OBSERVATION_COLUMNS,
        episodes=int(args.episodes),
    )
    dqn_train_positions = rollout_dqn_positions(prepared.train_frame, dqn_artifacts)
    dqn_positions = rollout_dqn_positions(prepared.test_frame, dqn_artifacts)
    dqn_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=dqn_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="dqn_rl_first_v2",
    )
    dqn_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=dqn_train_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["dqn_rl_first_v2"] = dqn_kelly
    dqn_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(dqn_positions["target_position"].tolist(), dqn_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="dqn_rl_first_v2_kelly",
    )
    comparison_rows.extend([dqn_result.metrics, dqn_kelly_result.metrics])

    ppo_artifacts = train_ppo_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_COMPACT_OBSERVATION_COLUMNS,
        total_timesteps=int(args.ppo_timesteps),
    )
    ppo_train_positions = rollout_ppo_positions(prepared.train_frame, ppo_artifacts)
    ppo_positions = rollout_ppo_positions(prepared.test_frame, ppo_artifacts)
    ppo_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=ppo_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="ppo_rl_first",
    )
    ppo_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=ppo_train_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["ppo_rl_first"] = ppo_kelly
    ppo_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(ppo_positions["target_position"].tolist(), ppo_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="ppo_rl_first_kelly",
    )
    comparison_rows.extend([ppo_result.metrics, ppo_kelly_result.metrics])

    meta_dqn_artifacts = train_meta_dqn_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_META_OBSERVATION_COLUMNS,
        episodes=int(args.episodes),
    )
    meta_dqn_train_positions = rollout_meta_dqn_positions(prepared.train_frame, meta_dqn_artifacts)
    meta_dqn_positions = rollout_meta_dqn_positions(prepared.test_frame, meta_dqn_artifacts)
    meta_dqn_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=meta_dqn_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="dqn_meta_controller",
    )
    meta_dqn_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=meta_dqn_train_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["dqn_meta_controller"] = meta_dqn_kelly
    meta_dqn_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(meta_dqn_positions["target_position"].tolist(), meta_dqn_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="dqn_meta_controller_kelly",
    )
    comparison_rows.extend([meta_dqn_result.metrics, meta_dqn_kelly_result.metrics])

    meta_ppo_artifacts = train_meta_ppo_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_META_OBSERVATION_COLUMNS,
        total_timesteps=int(args.ppo_timesteps),
    )
    meta_ppo_train_positions = rollout_meta_ppo_positions(prepared.train_frame, meta_ppo_artifacts)
    meta_ppo_positions = rollout_meta_ppo_positions(prepared.test_frame, meta_ppo_artifacts)
    meta_ppo_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=meta_ppo_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="ppo_meta_controller",
    )
    meta_ppo_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=meta_ppo_train_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["ppo_meta_controller"] = meta_ppo_kelly
    meta_ppo_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(meta_ppo_positions["target_position"].tolist(), meta_ppo_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="ppo_meta_controller_kelly",
    )
    comparison_rows.extend([meta_ppo_result.metrics, meta_ppo_kelly_result.metrics])

    binary_meta_dqn_artifacts = train_binary_meta_dqn_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_META_OBSERVATION_COLUMNS,
        episodes=int(args.episodes),
    )
    binary_meta_dqn_train_positions = rollout_binary_meta_dqn_positions(prepared.train_frame, binary_meta_dqn_artifacts)
    binary_meta_dqn_positions = rollout_binary_meta_dqn_positions(prepared.test_frame, binary_meta_dqn_artifacts)
    binary_meta_dqn_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=binary_meta_dqn_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="dqn_binary_meta_controller",
    )
    binary_meta_dqn_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=binary_meta_dqn_train_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["dqn_binary_meta_controller"] = binary_meta_dqn_kelly
    binary_meta_dqn_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(binary_meta_dqn_positions["target_position"].tolist(), binary_meta_dqn_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="dqn_binary_meta_controller_kelly",
    )
    comparison_rows.extend([binary_meta_dqn_result.metrics, binary_meta_dqn_kelly_result.metrics])

    binary_meta_ppo_artifacts = train_binary_meta_ppo_agent(
        train_frame=prepared.train_frame,
        observation_columns=RL_META_OBSERVATION_COLUMNS,
        total_timesteps=int(args.ppo_timesteps),
    )
    binary_meta_ppo_train_positions = rollout_binary_meta_ppo_positions(prepared.train_frame, binary_meta_ppo_artifacts)
    binary_meta_ppo_positions = rollout_binary_meta_ppo_positions(prepared.test_frame, binary_meta_ppo_artifacts)
    binary_meta_ppo_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=binary_meta_ppo_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="ppo_binary_meta_controller",
    )
    binary_meta_ppo_kelly = estimate_kelly_fraction(
        feature_frame=prepared.train_frame,
        positions=binary_meta_ppo_train_positions["target_position"].tolist(),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        max_fraction=kelly_max_fraction,
        fraction_multiplier=kelly_fraction_multiplier,
    )
    kelly_fractions["ppo_binary_meta_controller"] = binary_meta_ppo_kelly
    binary_meta_ppo_kelly_result = backtest_positions(
        feature_frame=prepared.test_frame,
        positions=apply_kelly_overlay(binary_meta_ppo_positions["target_position"].tolist(), binary_meta_ppo_kelly),
        fee_rate_per_rebalance=cfg.execution.fee_rate_per_rebalance,
        risk_penalty=cfg.execution.risk_penalty,
        initial_capital=cfg.research.initial_capital,
        strategy_name="ppo_binary_meta_controller_kelly",
    )
    comparison_rows.extend([binary_meta_ppo_result.metrics, binary_meta_ppo_kelly_result.metrics])

    report_root = (repo_root / args.report_dir).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = report_root / "comparison_metrics.csv"
    comparison_df.to_csv(comparison_path, index=False)

    summary = {
        "rows": {
            "raw": int(len(raw_frame)),
            "features": int(len(prepared.feature_frame)),
            "train": int(len(prepared.train_frame)),
            "test": int(len(prepared.test_frame)),
        },
        "gb_threshold": float(gb_threshold),
        "dqn_episode_reward_last_5": float(sum(dqn_artifacts.episode_rewards[-5:]) / max(1, len(dqn_artifacts.episode_rewards[-5:]))),
        "meta_dqn_episode_reward_last_5": float(
            sum(meta_dqn_artifacts.episode_rewards[-5:]) / max(1, len(meta_dqn_artifacts.episode_rewards[-5:]))
        ),
        "binary_meta_dqn_episode_reward_last_5": float(
            sum(binary_meta_dqn_artifacts.episode_rewards[-5:]) / max(1, len(binary_meta_dqn_artifacts.episode_rewards[-5:]))
        ),
        "kelly_fraction_multiplier": kelly_fraction_multiplier,
        "kelly_max_fraction": kelly_max_fraction,
        "kelly_fractions": kelly_fractions,
        "comparison_metrics_path": str(comparison_path),
        "trade_metrics": comparison_df.to_dict(orient="records"),
    }
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
