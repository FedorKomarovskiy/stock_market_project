from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simple_basis_rl.env import SimpleBasisEnv
from simple_basis_rl.positioning import build_base_neutral_position, validate_position
from simple_basis_rl.signals import add_basis_features


def _market_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
            "spot": [100.0, 101.0, 102.0, 103.0],
            "perp": [102.0, 102.5, 102.8, 103.2],
        }
    )


def test_basis_is_perp_minus_spot() -> None:
    df = add_basis_features(_market_df(), window=2)
    expected = df["perp"] - df["spot"]
    assert np.allclose(df["basis"].to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_zscore_is_computed_from_past_window() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
            "spot": [100.0, 100.0, 100.0, 100.0],
            "perp": [101.0, 102.0, 103.0, 104.0],
        }
    )
    feats = add_basis_features(df, window=2)
    # basis = [1,2,3,4], at idx=2 history window is [1,2]
    # mean=1.5, std=0.5, z=(3-1.5)/0.5 = 3
    assert feats.loc[2, "rolling_mean"] == pytest.approx(1.5)
    assert feats.loc[2, "rolling_std"] == pytest.approx(0.5)
    assert feats.loc[2, "basis_z"] == pytest.approx(3.0)


def test_position_is_base_neutral() -> None:
    pos = build_base_neutral_position(
        spot_price=100.0,
        perp_price=102.0,
        capital_usd=1000.0,
        contract_size=1.0,
        side="long_basis",
    )
    validate_position(pos)
    assert abs(pos["hedge_error"]) <= 1e-9
    assert pos["spot_qty"] == pytest.approx(pos["perp_base_qty"])


def test_long_basis_pnl_formula_is_correct() -> None:
    market = add_basis_features(_market_df(), window=2).dropna().reset_index(drop=True)
    env = SimpleBasisEnv(market, capital_usd=1000.0, contract_size=1.0)
    env.reset()
    _obs, reward, _terminated, _truncated, info = env.step(1)  # open_long_basis
    # First step uses rows 0->1 after dropna:
    # spot: 102 -> 103 (delta=+1), perp: 102.8 -> 103.2 (delta=+0.4)
    # spot leg uses exact qty, futures leg is rounded to whole contracts.
    expected_spot_qty = 1000.0 / 102.0
    expected_perp_base_qty = float(int(round(expected_spot_qty / 1.0))) * 1.0
    expected_pnl = expected_spot_qty * 1.0 - expected_perp_base_qty * 0.4
    assert reward == pytest.approx(expected_pnl)
    assert info["position"] == 1


def test_short_basis_pnl_formula_is_correct() -> None:
    market = add_basis_features(_market_df(), window=2).dropna().reset_index(drop=True)
    env = SimpleBasisEnv(market, capital_usd=1000.0, contract_size=1.0)
    env.reset()
    _obs, reward, _terminated, _truncated, info = env.step(2)  # open_short_basis
    expected_spot_qty = 1000.0 / 102.0
    expected_perp_base_qty = float(int(round(expected_spot_qty / 1.0))) * 1.0
    expected_pnl = -expected_spot_qty * 1.0 + expected_perp_base_qty * 0.4
    assert reward == pytest.approx(expected_pnl)
    assert info["position"] == -1


def test_env_step_opens_long_basis() -> None:
    market = add_basis_features(_market_df(), window=2).dropna().reset_index(drop=True)
    env = SimpleBasisEnv(market, capital_usd=1000.0, contract_size=1.0)
    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step(1)
    assert info["position"] == 1


def test_env_step_opens_short_basis() -> None:
    market = add_basis_features(_market_df(), window=2).dropna().reset_index(drop=True)
    env = SimpleBasisEnv(market, capital_usd=1000.0, contract_size=1.0)
    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step(2)
    assert info["position"] == -1


def test_env_step_closes_position() -> None:
    df = add_basis_features(_market_df(), window=2).dropna().reset_index(drop=True)
    # Need at least 3 rows for open then close on next step.
    extra = pd.DataFrame(
        {
            "timestamp": [df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)],
            "spot": [104.0],
            "perp": [103.5],
        }
    )
    df2 = pd.concat([_market_df(), extra], ignore_index=True)
    market = add_basis_features(df2, window=2).dropna().reset_index(drop=True)

    env = SimpleBasisEnv(market, capital_usd=1000.0, contract_size=1.0)
    env.reset()
    env.step(1)  # open
    _obs, _reward, _terminated, _truncated, info = env.step(3)  # close
    assert info["position"] == 0
