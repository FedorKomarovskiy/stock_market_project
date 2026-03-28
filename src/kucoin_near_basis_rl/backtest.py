from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import ACTION_TO_POSITION, BaselinePolicy, POSITION_TO_ACTION
from .qlearning import QLearningAgent, StateDiscretizer


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


@dataclass
class BacktestResult:
    history: pd.DataFrame
    metrics: dict[str, float | str]


def infer_periods_per_year(timestamps: pd.Series) -> float:
    ts = pd.to_datetime(timestamps, utc=True)
    if len(ts) < 2:
        return 1.0
    diffs = ts.sort_values().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 1.0
    interval_seconds = float(diffs.median())
    if interval_seconds <= 0:
        return 1.0
    return max(1.0, SECONDS_PER_YEAR / interval_seconds)


def build_baseline_positions(
    feature_frame: pd.DataFrame,
    enter_zscore: float,
    exit_zscore: float,
) -> pd.Series:
    policy = BaselinePolicy(enter_zscore=enter_zscore, exit_zscore=exit_zscore)
    current_position = 0
    positions: list[int] = []
    for row in feature_frame.itertuples():
        current_position = policy.decide_position(float(row.basis_zscore), current_position)
        positions.append(current_position)
    return pd.Series(positions, index=feature_frame.index, name="target_position", dtype="int64")


def rollout_rl_positions(
    feature_frame: pd.DataFrame,
    agent: QLearningAgent,
    discretizer: StateDiscretizer,
    observation_columns: list[str],
) -> pd.DataFrame:
    current_position = 0
    rows: list[dict[str, int | bool]] = []
    for row in feature_frame.itertuples():
        observation = np.array(
            [float(getattr(row, column)) for column in observation_columns] + [float(current_position)],
            dtype=np.float64,
        )
        state = discretizer.transform(observation)
        state_known = agent.has_state(state)
        action = agent.greedy_action(state) if state_known else POSITION_TO_ACTION[0]
        current_position = ACTION_TO_POSITION[action]
        rows.append(
            {
                "action": int(action),
                "state_known": bool(state_known),
                "target_position": int(current_position),
            }
        )
    return pd.DataFrame(rows, index=feature_frame.index)


def backtest_positions(
    feature_frame: pd.DataFrame,
    positions: Sequence[float] | pd.Series,
    fee_rate_per_rebalance: float,
    risk_penalty: float,
    initial_capital: float,
    strategy_name: str,
) -> BacktestResult:
    if len(feature_frame) < 2:
        raise ValueError("feature_frame must contain at least 2 rows for backtest.")
    if len(positions) != len(feature_frame):
        raise ValueError("positions length must match feature_frame length.")

    position_values = [float(value) for value in positions]
    equity = float(initial_capital)
    previous_position = 0.0
    rows: list[dict[str, float | int | str]] = []

    for idx in range(len(feature_frame) - 1):
        row = feature_frame.iloc[idx]
        next_row = feature_frame.iloc[idx + 1]
        target_position = position_values[idx]

        basis_change = float(next_row["basis"] - row["basis"])
        gross_return = target_position * basis_change
        turnover = abs(target_position - previous_position)
        fee_cost = float(fee_rate_per_rebalance) * turnover
        risk_cost = float(risk_penalty) * abs(target_position) * abs(float(row["basis_zscore"]))
        strategy_return = gross_return - fee_cost - risk_cost
        equity *= max(1e-9, 1.0 + strategy_return)

        rows.append(
            {
                "strategy": strategy_name,
                "decision_index": idx,
                "timestamp": next_row["timestamp"],
                "basis": float(next_row["basis"]),
                "basis_zscore": float(row["basis_zscore"]),
                "previous_position": float(previous_position),
                "target_position": float(target_position),
                "basis_change": basis_change,
                "gross_return": gross_return,
                "fee_cost": fee_cost,
                "risk_cost": risk_cost,
                "strategy_return": strategy_return,
                "equity": float(equity),
            }
        )
        previous_position = target_position

    history = pd.DataFrame(rows)
    return BacktestResult(
        history=history,
        metrics=calculate_strategy_metrics(
            history=history,
            initial_capital=initial_capital,
            strategy_name=strategy_name,
        ),
    )


def calculate_strategy_metrics(
    history: pd.DataFrame,
    initial_capital: float,
    strategy_name: str,
) -> dict[str, float | str]:
    if history.empty:
        return {
            "strategy": strategy_name,
            "final_equity": float(initial_capital),
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "annualized_volatility": 0.0,
            "turnover_events": 0.0,
            "exposure_ratio": 0.0,
            "hit_rate": 0.0,
        }

    equity = history["equity"].astype(float)
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_drawdown = float(drawdown.min())

    returns = history["strategy_return"].astype(float)
    periods_per_year = infer_periods_per_year(history["timestamp"])
    returns_std = float(returns.std(ddof=0))
    sharpe = 0.0
    if returns_std > 0:
        sharpe = float((returns.mean() / returns_std) * np.sqrt(periods_per_year))

    annualized_volatility = float(returns_std * np.sqrt(periods_per_year))
    total_return = float(equity.iloc[-1] / float(initial_capital) - 1.0)
    timestamp_series = pd.to_datetime(history["timestamp"], utc=True)
    start_ts = timestamp_series.iloc[0]
    end_ts = timestamp_series.iloc[-1]
    span_years = max((end_ts - start_ts).total_seconds() / SECONDS_PER_YEAR, 1.0 / periods_per_year)
    cagr = float((equity.iloc[-1] / float(initial_capital)) ** (1.0 / span_years) - 1.0)

    turnover_events = float((history["target_position"] != history["previous_position"]).sum())
    exposure_ratio = float((history["target_position"] != 0).mean())
    hit_rate = float((history["strategy_return"] > 0).mean())

    return {
        "strategy": strategy_name,
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annualized_volatility,
        "turnover_events": turnover_events,
        "exposure_ratio": exposure_ratio,
        "hit_rate": hit_rate,
    }


def estimate_kelly_fraction(
    feature_frame: pd.DataFrame,
    positions: Sequence[float] | pd.Series,
    fee_rate_per_rebalance: float,
    risk_penalty: float,
    max_fraction: float,
    fraction_multiplier: float = 0.5,
    min_active_rows: int = 24,
) -> float:
    if len(feature_frame) < 2 or len(positions) != len(feature_frame):
        return 0.0
    trial = backtest_positions(
        feature_frame=feature_frame,
        positions=positions,
        fee_rate_per_rebalance=fee_rate_per_rebalance,
        risk_penalty=risk_penalty,
        initial_capital=1.0,
        strategy_name="kelly_calibration",
    )
    history = trial.history
    active = history.loc[history["target_position"].abs() > 1e-9, "strategy_return"].astype(float)
    if len(active) < int(min_active_rows):
        return min(1.0, float(max_fraction))
    mean_return = float(active.mean())
    variance = float(active.var(ddof=0))
    if variance <= 1e-12 or mean_return <= 0.0:
        return 0.0
    raw_fraction = mean_return / variance
    bounded_fraction = raw_fraction / (1.0 + abs(raw_fraction))
    scaled_fraction = float(max_fraction) * float(fraction_multiplier) * bounded_fraction
    return float(np.clip(scaled_fraction, 0.0, float(max_fraction)))


def apply_kelly_overlay(
    positions: Sequence[float] | pd.Series,
    kelly_fraction: float,
) -> list[float]:
    scale = float(np.clip(kelly_fraction, 0.0, np.inf))
    return [float(value) * scale for value in positions]


def combine_backtest_results(
    results: Sequence[BacktestResult],
    initial_capital: float,
    strategy_name: str,
) -> BacktestResult:
    histories = [result.history.copy() for result in results if result.history is not None and not result.history.empty]
    if not histories:
        return BacktestResult(
            history=pd.DataFrame(),
            metrics=calculate_strategy_metrics(
                history=pd.DataFrame(),
                initial_capital=initial_capital,
                strategy_name=strategy_name,
            ),
        )

    combined = pd.concat(histories, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    equity = float(initial_capital)
    equities: list[float] = []
    for strategy_return in combined["strategy_return"].astype(float):
        equity *= max(1e-9, 1.0 + float(strategy_return))
        equities.append(float(equity))
    combined["equity"] = equities
    combined["strategy"] = strategy_name
    combined["decision_index"] = np.arange(len(combined), dtype=int)
    return BacktestResult(
        history=combined,
        metrics=calculate_strategy_metrics(
            history=combined,
            initial_capital=initial_capital,
            strategy_name=strategy_name,
        ),
    )
