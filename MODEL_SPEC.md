# Model Spec: NEAR Spot + Perpetual Basis RL

## 1. Instruments

- Spot: `NEAR-USDT`
- Futures perpetual: `NEARUSDTM`
- Exchange: KuCoin
- Research timeframe: `1 hour`
- Live profile in repository: `1 minute`

## 2. Market variable

The project trades the normalized basis:

```text
basis_t = (F_t - S_t) / S_t
```

where:
- `F_t` is futures close
- `S_t` is spot close

The strategy assumes short-horizon mean reversion of the basis after extreme deviations.

## 3. Feature engineering

Computed columns in the research pipeline:
- `basis`
- `spot_return`
- `futures_return`
- `basis_return`
- `basis_zscore`
- `spot_volatility`
- `futures_volatility`
- `basis_volatility`
- `volume_imbalance`
- `basis_momentum`
- `rolling_correlation`
- `spot_rsi`
- `futures_rsi`
- `basis_ema_fast`
- `basis_ema_slow`
- `basis_ema_gap`

## 4. Baseline policy

Z-score mean reversion:
- if `basis_zscore >= 3.0` -> short basis
- if `basis_zscore <= -3.0` -> long basis
- if `abs(basis_zscore) <= 0.5` -> flat
- otherwise keep previous position

Position semantics:
- `position = -1` -> short basis -> `SELL futures`, `BUY spot`
- `position = 0` -> flat
- `position = +1` -> long basis -> `BUY futures`, `SELL spot`

## 5. RL policy

Algorithm:
- tabular Q-learning

State:
- `basis`
- `spot_return`
- `futures_return`
- `basis_return`
- `basis_zscore`
- `spot_volatility`
- `futures_volatility`
- `basis_volatility`
- `volume_imbalance`
- `basis_momentum`
- `rolling_correlation`
- `spot_rsi`
- `futures_rsi`
- `basis_ema_fast`
- `basis_ema_slow`
- `basis_ema_gap`
- `current_position`

Action space:
- `0 -> short basis`
- `1 -> flat`
- `2 -> long basis`

Important implementation note:
- the RL state now uses the full engineered feature vector plus `current_position`
- unseen states in backtest/live default to `flat` instead of random action selection

## 6. Reward

For step `t -> t+1`:

```text
gross_return = target_position * (basis_{t+1} - basis_t)
fee_cost = fee_rate_per_rebalance * abs(target_position - previous_position)
risk_cost = risk_penalty * abs(target_position) * abs(basis_zscore_t)
reward = gross_return - fee_cost - risk_cost
```

Research configuration used in the final run:
- `fee_rate_per_rebalance = 0.0004`
- `risk_penalty = 0.00005`

## 7. Train / test protocol

Historical download window:
- `2024-01-01 00:00:00 UTC` to `2026-03-01 00:00:00 UTC`

After rolling-window feature preparation:
- train split: `2024-08-08 01:00:00 UTC` to `2025-09-10 22:00:00 UTC`
- test split: `2025-09-10 23:00:00 UTC` to `2026-02-28 23:00:00 UTC`

Split ratio:
- `70% train / 30% test`

## 8. Backtest metrics from the saved report

Out-of-sample results:

| Strategy | Total Return | CAGR | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: |
| Baseline z-score | 3.77% | 8.23% | 4.10 | -0.36% |
| RL agent | 2.77% | 6.00% | 4.06 | -0.13% |

## 9. Live deployment notes

- Live/shadow entrypoint: `run_trade_signal.py`
- Research entrypoint: `run_research_pipeline.py`
- For cash spot accounts, `allow_spot_short=false` is kept in the live config.
- Full symmetric live trading for the spot/perpetual pair requires spot shorting via margin or a modified execution layer.
