# KuCoin Delta-Neutral RL Project

Reproducible course project for crypto trading on KuCoin with:
- chosen pair: `NEAR-USDT` spot + `NEARUSDTM` perpetual futures
- baseline: z-score mean reversion on basis
- RL agent: tabular Q-learning
- outputs: historical data, engineered features, backtests, metrics, equity curve, notebook, live/shadow scripts

The repository is split into two layers:
- `research` layer for the course project and backtesting
- `live/shadow` layer for KuCoin deployment through PowerShell / bash

## 1. What is inside

Project requirements covered:
- download of historical OHLCV data for more than 2 years
- feature engineering: returns, volatility, volume imbalance, rolling correlation, RSI, EMA-gap
- baseline model: z-score basis strategy
- RL formalization: `state = engineered_features + current_position`, `action in {-1, 0, +1}`
- train/test time split without leakage
- backtest with `Sharpe`, `Max Drawdown`, `CAGR`
- equity-curve and market/position charts
- notebook and slide deck for presentation
- PowerShell / bash commands for live and shadow runs

## 2. Repository structure

- `run_research_pipeline.py` - end-to-end research pipeline for the course project
- `run_gb_forecast.py` - gradient boosting price forecast on the last 3 years of history
- `run_trade_signal.py` - launcher for `train / shadow / live`
- `trade_signal_executor_kucoin.py` - execution entrypoint
- `config/project_near_hourly.json` - research config used for the final backtest
- `config/micro_near_v1_1m.json` - live/shadow minute profile
- `src/kucoin_near_basis_rl/features.py` - data preparation and engineered features
- `src/kucoin_near_basis_rl/train.py` - RL training
- `src/kucoin_near_basis_rl/backtest.py` - out-of-sample backtesting and metrics
- `src/kucoin_near_basis_rl/live.py` - paper/live decision loop
- `notebooks/project_near_basis_rl.ipynb` - reproducible notebook
- `presentation/project_slides.html` - slide deck source
- `presentation/project_slides.pdf` - exported PDF slides

## 3. Install

Core CLI environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements-core.txt
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements-core.txt
```

Optional notebook environment:

```powershell
python -m pip install jupyter ipykernel
```

Note:
- `requirements-core.txt` is enough for training, backtesting, tests and live/shadow scripts.
- `requirements.txt` additionally contains Jupyter-related packages. On some Windows setups full Jupyter installation may require long-path support.

## 4. Reproduce the course experiment

PowerShell:

```powershell
.\scripts\bot.ps1 -Action research
```

Equivalent direct command:

```powershell
python run_research_pipeline.py `
  --config config/project_near_hourly.json `
  --model-out models/project_near_hourly_qlearning.json `
  --report-dir reports/project_near_hourly `
  --raw-out data/project_near_hourly_raw.csv `
  --features-out reports/project_near_hourly/features.csv `
  --start "2024-01-01T00:00:00Z" `
  --end "2026-03-01T00:00:00Z"
```

bash:

```bash
./scripts/bot.sh research
```

The pipeline performs:
1. downloads and merges KuCoin spot/futures history
2. builds features
3. splits data into `70% train / 30% test`
4. trains the Q-learning agent
5. backtests baseline and RL on the test split
6. saves metrics and plots

Main outputs:
- `data/project_near_hourly_raw.csv`
- `models/project_near_hourly_qlearning.json`
- `reports/project_near_hourly/metrics.csv`
- `reports/project_near_hourly/equity_curve.png`
- `reports/project_near_hourly/market_and_positions.png`
- `reports/project_near_hourly/summary.json`

## 5. Final backtest snapshot

Latest reproducible run in this repository:
- download window: `2024-01-01 00:00:00 UTC` to `2026-03-01 00:00:00 UTC`
- feature frame after rolling windows: `2024-08-08 01:00:00 UTC` to `2026-02-28 23:00:00 UTC`
- train split: `2024-08-08 01:00:00 UTC` to `2025-09-10 22:00:00 UTC`
- test split: `2025-09-10 23:00:00 UTC` to `2026-02-28 23:00:00 UTC`
- raw rows: `13,739`
- feature rows: `13,666`

Backtest assumptions:
- hourly bars
- combined rebalance fee assumption in research config: `0.0004`
- additional risk penalty in reward/backtest: `0.00005 * |position| * |zscore|`
- initial capital: `10,000 USDT`

Results on the test split:

| Strategy | Total Return | CAGR | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: |
| Baseline z-score | 3.77% | 8.23% | 4.10 | -0.36% |
| RL agent | 2.77% | 6.00% | 4.06 | -0.13% |

Interpretation:
- the baseline is slightly more profitable on this split
- the RL agent is also profitable, trades less, and shows lower drawdown
- this makes the baseline a strong reference model and the RL policy a smoother alternative

## 6. State, action and reward

State used by the RL agent:
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

Actions:
- `0 -> short basis`
- `1 -> flat`
- `2 -> long basis`

Reward per step:
- `position * delta(basis)`
- minus rebalance fee
- minus risk penalty for holding exposure under extreme z-score

Gradient boosting forecast:
- trains two boosting models on the same engineered dataset
- targets: next-bar `spot_close` and next-bar `futures_close`
- default download window: last 3 years from the current UTC timestamp

Run GB forecast:

```bash
python run_gb_forecast.py
```

## 7. Notebook and presentation

Open the notebook:

```powershell
python -m jupyter lab notebooks/project_near_basis_rl.ipynb
```

Slides already included:
- `presentation/project_slides.html`
- `presentation/project_slides.pdf`

To regenerate the PDF on Windows with Microsoft Edge:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\presentation\export_presentation.ps1
```

## 8. KuCoin live and shadow mode

Create local env file:

```powershell
.\scripts\bot.ps1 -Action env-template
```

Fill `.runtime/project.env`:

```env
KUCOIN_API_KEY=...
KUCOIN_API_SECRET=...
KUCOIN_API_PASSPHRASE=...
FINNHUB_API_KEY=...
CRYPTOPANIC_API_KEY=...
X_BEARER_TOKEN=...
X_CONSUMER_KEY=...
X_CONSUMER_SECRET=...
```

Shadow once:

```powershell
python run_trade_signal.py --mode train --config config/micro_near_v1_1m.json --model-path models/near_basis_qlearning.json
python run_trade_signal.py --mode shadow --once --config config/micro_near_v1_1m.json --model-path models/near_basis_qlearning.json
```

Live:

```powershell
python run_trade_signal.py --mode live --run-real-order --config config/micro_near_v1_1m.json --model-path models/near_basis_qlearning.json
```

Important live note:
- `config/micro_near_v1_1m.json` is the lecture-style live profile.
- for cash spot accounts the project keeps `allow_spot_short=false`; in that case states requiring a spot short are flattened.
- full two-sided live deployment for a spot/perpetual pair requires either margin-enabled spot shorting or a modified execution layer.

## 9. Tests

```powershell
$env:PYTHONPATH="src"
python -m pytest tests -q
```

Current status in local verification:
- `16 passed`

## 10. GitHub checklist

Before submission:
1. create a private repository
2. push this project
3. add collaborators: instructor, assistants, and your teammate
4. keep `.runtime/project.env` private

Minimal git commands:

```bash
git init
git add .
git commit -m "Add reproducible KuCoin RL project"
git branch -M main
git remote add origin https://github.com/<your_user>/<your_private_repo>.git
git push -u origin main
```
