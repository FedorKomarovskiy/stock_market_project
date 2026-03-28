from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult, backtest_positions, build_baseline_positions, rollout_rl_positions
from .config import AppConfig, load_config
from .features import build_feature_frame
from .finnhub_news import build_news_feature_frame
from .kucoin_api import KuCoinPublicDataClient
from .qlearning import save_model_artifact
from .train import train_agent_from_features


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_raw_frame(
    cfg: AppConfig,
    source_csv: str | None,
    start_iso: str | None,
    end_iso: str | None,
) -> pd.DataFrame:
    if source_csv:
        frame = pd.read_csv(source_csv, parse_dates=["timestamp"])
        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize("UTC")
        return frame

    data_client = KuCoinPublicDataClient(cfg.api)
    if start_iso and end_iso:
        start_dt = _parse_dt(start_iso)
        end_dt = _parse_dt(end_iso)
    else:
        start_dt, end_dt = data_client.utc_lookback(cfg.data.lookback_minutes)
    return data_client.fetch_merged_candles(cfg.data, start_dt=start_dt, end_dt=end_dt)


def _time_split(feature_frame: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fraction = float(cfg.research.train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("research.train_fraction must be between 0 and 1.")

    split_index = int(len(feature_frame) * train_fraction)
    split_index = max(split_index, int(cfg.research.min_train_rows))
    split_index = min(split_index, len(feature_frame) - int(cfg.research.min_test_rows))
    if split_index <= 0 or split_index >= len(feature_frame):
        raise RuntimeError("Unable to create train/test split with current research settings.")

    train_frame = feature_frame.iloc[:split_index].reset_index(drop=True)
    test_frame = feature_frame.iloc[split_index:].reset_index(drop=True)
    if len(train_frame) < int(cfg.research.min_train_rows):
        raise RuntimeError("Train split is too small for research backtest.")
    if len(test_frame) < int(cfg.research.min_test_rows):
        raise RuntimeError("Test split is too small for research backtest.")
    return train_frame, test_frame


def _plot_equity_curves(
    baseline_result: BacktestResult,
    rl_result: BacktestResult,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        baseline_result.history["timestamp"],
        baseline_result.history["equity"],
        label="Baseline z-score",
        linewidth=2,
    )
    ax.plot(
        rl_result.history["timestamp"],
        rl_result.history["equity"],
        label="RL agent",
        linewidth=2,
    )
    ax.set_title("Equity Curve on Test Split")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_market_and_positions(
    test_frame: pd.DataFrame,
    baseline_positions: pd.Series,
    rl_positions: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(test_frame["timestamp"], test_frame["basis"], color="tab:blue", linewidth=1.6)
    axes[0].set_title("Basis")
    axes[0].grid(alpha=0.3)

    axes[1].plot(test_frame["timestamp"], test_frame["basis_zscore"], color="tab:red", linewidth=1.4)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Basis Z-score")
    axes[1].grid(alpha=0.3)

    axes[2].step(
        test_frame["timestamp"],
        baseline_positions.to_numpy(dtype=float),
        where="post",
        label="Baseline position",
        linewidth=1.6,
    )
    axes[2].step(
        test_frame["timestamp"],
        rl_positions["target_position"].to_numpy(dtype=float),
        where="post",
        label="RL position",
        linewidth=1.6,
    )
    axes[2].set_title("Target Positions")
    axes[2].set_yticks([-1, 0, 1])
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_research_experiment(
    config_path: str,
    model_out: str,
    report_dir: str,
    source_csv: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    raw_out: str | None = None,
    features_out: str | None = None,
    episodes_override: int | None = None,
) -> dict[str, object]:
    cfg = load_config(config_path)
    if episodes_override is not None:
        if episodes_override <= 0:
            raise ValueError("episodes_override must be > 0")
        cfg.rl.episodes = int(episodes_override)

    raw_frame = _load_raw_frame(cfg, source_csv=source_csv, start_iso=start_iso, end_iso=end_iso)
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    if raw_out:
        raw_path = Path(raw_out)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_frame.to_csv(raw_path, index=False)

    news_feature_frame = build_news_feature_frame(raw_frame, cfg.news)
    feature_frame = build_feature_frame(raw_frame, cfg.features, news_feature_frame=news_feature_frame)
    train_frame, test_frame = _time_split(feature_frame, cfg)
    split_feature_frame = feature_frame.copy()
    split_feature_frame["split"] = ["train"] * len(train_frame) + ["test"] * len(test_frame)
    if features_out:
        features_path = Path(features_out)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        split_feature_frame.to_csv(features_path, index=False)

    artifacts = train_agent_from_features(train_frame, cfg)
    save_model_artifact(
        path=model_out,
        agent=artifacts.agent,
        discretizer=artifacts.discretizer,
        observation_columns=artifacts.observation_columns,
        metadata={
            "config_path": str(config_path),
            "rows_used": int(artifacts.rows_used),
            "train_start": str(train_frame.iloc[0]["timestamp"]),
            "train_end": str(train_frame.iloc[-1]["timestamp"]),
            "test_start": str(test_frame.iloc[0]["timestamp"]),
            "test_end": str(test_frame.iloc[-1]["timestamp"]),
        },
    )

    baseline_positions = build_baseline_positions(
        test_frame,
        enter_zscore=cfg.baseline.enter_zscore,
        exit_zscore=cfg.baseline.exit_zscore,
    )
    rl_positions = rollout_rl_positions(
        test_frame,
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

    rl_result.history = rl_result.history.merge(
        rl_positions.iloc[:-1].reset_index(drop=True),
        left_on="decision_index",
        right_index=True,
        how="left",
    )

    metrics_df = pd.DataFrame([baseline_result.metrics, rl_result.metrics])
    metrics_path = report_root / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    baseline_history_path = report_root / "baseline_backtest.csv"
    rl_history_path = report_root / "rl_backtest.csv"
    baseline_result.history.to_csv(baseline_history_path, index=False)
    rl_result.history.to_csv(rl_history_path, index=False)

    equity_curve_path = report_root / "equity_curve.png"
    market_plot_path = report_root / "market_and_positions.png"
    _plot_equity_curves(baseline_result, rl_result, equity_curve_path)
    _plot_market_and_positions(test_frame, baseline_positions, rl_positions, market_plot_path)

    summary = {
        "pair": {
            "spot_symbol": cfg.data.spot_symbol,
            "futures_symbol": cfg.data.futures_symbol,
        },
        "rows": {
            "raw": int(len(raw_frame)),
            "features": int(len(feature_frame)),
            "train": int(len(train_frame)),
            "test": int(len(test_frame)),
        },
        "model_path": str(Path(model_out).resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "baseline_backtest_path": str(baseline_history_path.resolve()),
        "rl_backtest_path": str(rl_history_path.resolve()),
        "equity_curve_path": str(equity_curve_path.resolve()),
        "market_plot_path": str(market_plot_path.resolve()),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary
