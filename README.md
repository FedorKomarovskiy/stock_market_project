# Stock Market Project

## Notebook for review

Main submission notebook:

- `notebooks/stock_market_volatility_submission.ipynb`

The notebook contains the full end-to-end solution in one place:

- step-by-step comments for each stage;
- forecast vs realized volatility plots;
- model visualizations;
- metric tables including `Sharpe` and `Max Drawdown`;
- Monte Carlo volatility simulation.

The original repository modules are still available below, but the notebook above is the primary file for review.

## Original repository structure

Research and execution repository for basis trading on KuCoin:
- spot: `NEAR-USDT`
- perpetual: `NEARUSDTM`
- market variable: normalized basis `(F - S) / S`

The repository started as a course project with a baseline z-score strategy and tabular Q-learning. It now includes:
- a unified market + news feature pipeline
- baseline, gradient boosting, CatBoost and ensemble research pipelines
- several RL variants, including DQN, PPO, meta-controllers and an enhanced `Double DQN + Dueling + Prioritized Replay` controller
- walk-forward evaluation for more objective comparison
- shadow/live execution scripts for KuCoin

## 1. Current architecture

The project is split into five layers.

### Data layer
- KuCoin market data: `src/kucoin_near_basis_rl/kucoin_api.py`
- News aggregation and caching: `src/kucoin_near_basis_rl/finnhub_news.py`
- Supported news sources: `Finnhub + CryptoPanic + X`

### Feature and signal layer
- Core market features: `src/kucoin_near_basis_rl/features.py`
- Enriched feature pipeline: `src/kucoin_near_basis_rl/feature_pipeline.py`
- Baseline signal: `src/kucoin_near_basis_rl/baseline.py`
- GB auxiliary signal: `src/kucoin_near_basis_rl/gb_model.py`

### Model layer
- Original tabular RL: `src/kucoin_near_basis_rl/qlearning.py`
- DQN: `src/kucoin_near_basis_rl/dqn_agent.py`
- PPO: `src/kucoin_near_basis_rl/ppo_agent.py`
- Binary / ternary meta-controllers: `src/kucoin_near_basis_rl/meta_controller_agent.py`
- Enhanced DQN meta-controller: `src/kucoin_near_basis_rl/enhanced_meta_dqn.py`
- CatBoost models: `src/kucoin_near_basis_rl/catboost_model.py`, `src/kucoin_near_basis_rl/catboost_classifier_model.py`
- Ensemble research model: `src/kucoin_near_basis_rl/ensemble_model.py`

### Evaluation layer
- Research experiment and plots: `src/kucoin_near_basis_rl/research.py`
- Backtest engine and metrics: `src/kucoin_near_basis_rl/backtest.py`
- Model comparison runners:
  - `run_model_comparison.py`
  - `run_rl_strategy_comparison.py`
  - `run_top5_dqn_walkforward.py`

### Execution layer
- Cross-platform launcher: `run_trade_signal.py`
- Train / shadow / live executor: `trade_signal_executor_kucoin.py`
- Live decision loop: `src/kucoin_near_basis_rl/live.py`

## 2. Features

### Original market features
The original project already used basis and market microstructure features:
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

### Added news features
The current repository adds multi-source news features over `6h` and `24h` windows:
- `news_count_*`
- `news_sentiment_mean_*`
- `news_weighted_sentiment_mean_*`
- `news_sentiment_sum_*`
- `news_sentiment_abs_sum_*`
- `news_relevance_mean_*`
- `news_near_count_*`
- `news_positive_count_*`
- `news_negative_count_*`
- `news_cryptopanic_count_*`
- `news_finnhub_count_*`
- `news_x_count_*`
- `news_source_diversity_*`
- `news_burst_ratio_6h_24h`

### Added baseline-derived features
- `baseline_position`
- `baseline_enter_long_flag`
- `baseline_enter_short_flag`
- `baseline_flat_flag`
- `distance_to_baseline_entry`
- `distance_to_baseline_exit`

### Added GB-derived features
- `gb_pred_basis_change`
- `gb_pred_abs_basis_change`
- `gb_pred_direction`
- `gb_high_confidence_flag`
- `gb_confidence_score`

### Added meta-signal features for RL
- `rl_regime_active_flag`
- `meta_signal_direction`
- `meta_signal_strength`
- `meta_signal_alignment_flag`

### Added stateful RL features
The enhanced DQN also uses portfolio-state features:
- `current_position`
- `position_abs`
- `bars_in_trade_scaled`
- `entry_basis_zscore_scaled`
- `unrealized_pnl_scaled`
- `time_since_last_trade_scaled`
- `recent_turnover_rate`

## 3. Baseline model

`baseline` is the original strategy of the project, not a separate later addition.

It is a simple z-score mean-reversion policy:
- if `basis_zscore >= enter_zscore` -> `short basis`
- if `basis_zscore <= -enter_zscore` -> `long basis`
- if `abs(basis_zscore) <= exit_zscore` -> `flat`
- otherwise keep the previous position

In the current repository the baseline plays three roles:
- benchmark strategy
- source of engineered features
- one of the two components of the `meta-signal` together with GB

## 4. What the RL layer does now

There are several RL variants in the repository. The current RL direction is not “predict the market from scratch”, but `RL as final decision-maker over structured signals`.

The strongest RL version at the moment is:
- `dqn_enhanced_meta_controller`

It receives:
- compact market features
- news features
- baseline features
- GB predictions and confidence
- meta-signal features
- stateful position / PnL context

Its action space is:
- `0 = skip`
- `1 = half-size`
- `2 = full-size`

The actual trade direction comes from the sign of `meta_signal_direction`, while RL decides whether the signal should be ignored, taken conservatively, or taken fully.

## 5. Install

### Core environment
macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements-core.txt
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements-core.txt
```

### Optional notebook environment

```bash
python -m pip install jupyter ipykernel
```

## 6. Unified credentials file

The repository now uses a single local runtime credentials file:

```text
.runtime/project.env
```

Create it from the template:

```bash
cp examples/project.env.example .runtime/project.env
```

or on Windows:

```powershell
.\scripts\bot.ps1 -Action env-template
```

Example:

```env
KUCOIN_API_KEY=...
KUCOIN_API_SECRET=...
KUCOIN_API_PASSPHRASE=...
KUCOIN_KEY_VERSION=2

FINNHUB_API_KEY=...
CRYPTOPANIC_API_KEY=...
X_BEARER_TOKEN=...
X_CONSUMER_KEY=...
X_CONSUMER_SECRET=...
```

Notes:
- all main runners automatically load `.runtime/project.env`
- `.runtime/kucoin.env` is still supported as a fallback for backward compatibility
- keep `.runtime/project.env` private and out of git

## 7. Main entrypoints

### Reproduce the original research pipeline

```bash
python run_research_pipeline.py \
  --config config/project_near_hourly.json \
  --model-out models/project_near_hourly_qlearning.json \
  --report-dir reports/project_near_hourly \
  --raw-out data/project_near_hourly_raw.csv \
  --features-out reports/project_near_hourly/features.csv \
  --start 2024-01-01T00:00:00Z \
  --end 2026-03-01T00:00:00Z
```

### Train gradient boosting forecast models

```bash
python run_gb_forecast.py \
  --config config/project_near_hourly.json \
  --report-dir reports/project_near_hourly_gb
```

### Compare forecast / ML families

```bash
python run_model_comparison.py \
  --config config/project_near_hourly.json \
  --source-csv data/project_near_hourly_raw.csv \
  --report-dir reports/model_comparison
```

### Compare RL and non-RL strategies on a recent window

```bash
python run_rl_strategy_comparison.py \
  --config config/project_near_hourly.json \
  --start 2026-01-15T00:00:00Z \
  --end 2026-03-27T00:00:00Z \
  --report-dir reports/rl_first_comparison_kelly_newsfull
```

### Objective walk-forward comparison of top strategies

```bash
python run_top5_dqn_walkforward.py \
  --config config/project_near_hourly.json \
  --start 2026-01-15T00:00:00Z \
  --end 2026-03-27T00:00:00Z \
  --report-dir reports/top5_dqn_walkforward_newsfull_tuned \
  --episodes 45 \
  --folds 2 \
  --reward-horizon 6
```

### Shadow / live trading

Train:

```bash
python run_trade_signal.py \
  --mode train \
  --config config/micro_near_v1_1m.json \
  --model-path models/near_basis_qlearning.json
```

Shadow once:

```bash
python run_trade_signal.py \
  --mode shadow \
  --once \
  --config config/micro_near_v1_1m.json \
  --model-path models/near_basis_qlearning.json
```

Live:

```bash
python run_trade_signal.py \
  --mode live \
  --run-real-order \
  --config config/micro_near_v1_1m.json \
  --model-path models/near_basis_qlearning.json
```

## 8. Latest experiment summary

### A. Broad recent-window comparison with the new news layer
Report:
- `reports/rl_first_comparison_kelly_newsfull/comparison_metrics.csv`
- window: `2026-01-15 -> 2026-03-27`
- evaluation: single holdout split on the recent news-enabled window

The table below shows the non-Kelly strategies from the latest broad comparison:

| Strategy | Total Return | Sharpe | Max Drawdown | Comment |
| --- | ---: | ---: | ---: | --- |
| `gradient_boosting` | 10.09% | 7.14 | -0.31% | Best absolute result |
| `dqn_binary_meta_controller` | 6.19% | 3.27 | -1.48% | Best RL result on this window |
| `gb_plus_baseline_filter` | 4.17% | 5.01 | -0.14% | Strong filtered non-RL benchmark |
| `baseline_zscore` | 3.77% | 4.10 | -0.36% | Original project strategy |
| `ppo_binary_meta_controller` | 1.35% | 0.88 | -1.54% | Positive but clearly weaker |
| `tabular_rl` | 0.15% | 0.81 | -0.12% | Original RL baseline |
| `dqn_meta_controller` | -0.38% | -0.20 | -2.01% | Underperforms |
| `ppo_meta_controller` | -0.37% | -0.26 | -2.44% | Underperforms |
| `ppo_rl_first` | -1.18% | -0.64 | -2.88% | Worse than meta-controller |
| `dqn_rl_first_v2` | -5.90% | -6.06 | -6.10% | Worst among recent RL variants |

Interpretation:
- on the recent news-enabled holdout, `GB` still dominates
- the best RL family member is the binary DQN meta-controller
- the original tabular RL remains mainly a baseline reference

### B. Latest stable walk-forward top-5 comparison
Report:
- `reports/top5_dqn_walkforward_newsfull_tuned/comparison_metrics_top5.csv`
- `reports/top5_dqn_walkforward_newsfull_tuned/summary.json`
- window: `2026-01-15 -> 2026-03-27`
- evaluation: `2-fold expanding walk-forward`

This is the more objective comparison because the models are retrained on each fold and tested on unseen future slices.

| Strategy | Total Return | Sharpe | Max Drawdown | Turnover |
| --- | ---: | ---: | ---: | ---: |
| `gradient_boosting` | 1.40% | 8.04 | -0.15% | 75 |
| `gb_plus_baseline_filter` | 0.59% | 6.48 | -0.04% | 10 |
| `gb_plus_baseline_filter_kelly` | 0.59% | 6.48 | -0.04% | 10 |
| `gradient_boosting_kelly` | 0.35% | 8.04 | -0.04% | 75 |
| `dqn_enhanced_meta_controller` | 0.19% | 3.58 | -0.04% | 6 |

Interpretation:
- `GB` is still the strongest model under walk-forward evaluation
- `enhanced DQN meta-controller` is the strongest stable RL candidate
- the enhanced DQN is profitable and materially better than the original RL baselines, but still weaker than `GB`
- Kelly overlay did not improve the walk-forward result materially

## 9. Reproducibility notes

What makes the current repository reproducible:
- single runtime credentials file: `.runtime/project.env`
- deterministic runners for research and comparisons
- tests for the pipeline and env loading
- saved reports and summaries in `reports/`
- walk-forward evaluation available in addition to ordinary backtest

Important distinction:
- `backtest` / holdout results are historical estimates
- `walk-forward` is a stricter historical evaluation
- `shadow` runs the strategy on new market data without real orders
- only `live` gives the real deposit result

## 10. Tests

```bash
PYTHONPATH=src python -m pytest tests/test_project_pipeline.py tests/test_kucoin_near_basis_rl.py
```

Current local status:
- `16 passed`

## 11. Useful helpers

PowerShell helper:

```powershell
.\scripts\bot.ps1 -Action research
.\scripts\bot.ps1 -Action shadow-once
.\scripts\bot.ps1 -Action live
```

bash helper:

```bash
./scripts/bot.sh research
./scripts/bot.sh shadow-once
./scripts/bot.sh live
```

## 12. Repository hygiene

Before pushing:
- do not commit `.runtime/project.env`
- do not commit local IDE files
- do not commit transient training artifacts

The repository already ignores:
- `.runtime/`
- `.idea/`
- `reports/`
- `models/`
- `catboost_info/`
