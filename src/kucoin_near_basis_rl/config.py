from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ApiConfig:
    api_key_env: str = "KUCOIN_API_KEY"
    api_secret_env: str = "KUCOIN_API_SECRET"
    api_passphrase_env: str = "KUCOIN_API_PASSPHRASE"
    is_sandbox: bool = False
    spot_base_url: str = "https://api.kucoin.com"
    futures_base_url: str = "https://api-futures.kucoin.com"


@dataclass
class DataConfig:
    spot_symbol: str = "NEAR-USDT"
    futures_symbol: str = "NEARUSDTM"
    interval: str = "1min"
    futures_granularity_minutes: int = 1
    lookback_minutes: int = 6_000


@dataclass
class FeatureConfig:
    zscore_window: int = 120
    volatility_window: int = 60
    volume_window: int = 30
    basis_momentum_lag: int = 5
    correlation_window: int = 60
    rsi_window: int = 14
    ema_fast_window: int = 12
    ema_slow_window: int = 26


@dataclass
class NewsConfig:
    enabled: bool = True
    use_finnhub: bool = True
    use_cryptopanic: bool = True
    use_x: bool = True
    api_key_env: str = "FINNHUB_API_KEY"
    cryptopanic_api_key_env: str = "CRYPTOPANIC_API_KEY"
    x_bearer_env: str = "X_BEARER_TOKEN"
    x_consumer_key_env: str = "X_CONSUMER_KEY"
    x_consumer_secret_env: str = "X_CONSUMER_SECRET"
    category: str = "crypto"
    x_query: str = '(NEAR OR "Near Protocol") lang:en -is:retweet'
    x_max_results: int = 25
    short_window_hours: int = 6
    long_window_hours: int = 24
    cache_path: str = ".runtime/finnhub_crypto_news.csv"
    request_timeout_sec: int = 20
    cache_retention_days: int = 30


@dataclass
class BaselineConfig:
    enter_zscore: float = 1.8
    exit_zscore: float = 0.35


@dataclass
class RlConfig:
    episodes: int = 80
    alpha: float = 0.08
    gamma: float = 0.98
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.97
    use_baseline_guidance: bool = False
    imitation_start: float = 0.35
    imitation_end: float = 0.05
    imitation_decay: float = 0.97
    baseline_bonus: float = 0.00008
    max_steps_per_episode: int = 2_000
    quantile_bins: list[float] = field(
        default_factory=lambda: [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
    )


@dataclass
class ExecutionConfig:
    quote_notional_usdt: float = 60.0
    futures_contract_multiplier: float = 1.0
    fee_rate_per_rebalance: float = 0.0012
    risk_penalty: float = 0.0002
    leverage: int = 2
    poll_seconds: int = 60
    default_paper_mode: bool = True
    allow_spot_short: bool = False


@dataclass
class ResearchConfig:
    train_fraction: float = 0.7
    initial_capital: float = 10_000.0
    min_train_rows: int = 500
    min_test_rows: int = 200


@dataclass
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    rl: RlConfig = field(default_factory=RlConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        return cls(
            api=ApiConfig(**payload.get("api", {})),
            data=DataConfig(**payload.get("data", {})),
            features=FeatureConfig(**payload.get("features", {})),
            news=NewsConfig(**payload.get("news", {})),
            baseline=BaselineConfig(**payload.get("baseline", {})),
            rl=RlConfig(**payload.get("rl", {})),
            execution=ExecutionConfig(**payload.get("execution", {})),
            research=ResearchConfig(**payload.get("research", {})),
        )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.from_dict(payload)
