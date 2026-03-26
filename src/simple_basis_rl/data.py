from __future__ import annotations

from typing import Optional

import pandas as pd

from delta_bot.kucoin_client import KuCoinRestClient

from .signals import add_basis_features


def _spot_candle_type(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported spot timeframe: {timeframe}")
    return mapping[tf]


def _perp_granularity(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported perp timeframe: {timeframe}")
    return mapping[tf]


def _normalize_perp_ts(ts: int | float | str) -> int:
    out = int(float(ts))
    if out > 10**12:
        out //= 1000
    return out


def fetch_spot_history(
    *,
    client: Optional[KuCoinRestClient] = None,
    symbol: str = "NEAR-USDT",
    timeframe: str = "1m",
    lookback_bars: int = 500,
) -> pd.DataFrame:
    c = client or KuCoinRestClient.from_env()
    raw = c.get_spot_candles(symbol=symbol, candle_type=_spot_candle_type(timeframe))
    rows = []
    for item in raw:
        try:
            ts = int(float(item[0]))
            close = float(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        rows.append({"timestamp": pd.to_datetime(ts, unit="s", utc=True), "spot": close})

    out = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    if lookback_bars > 0:
        out = out.tail(int(lookback_bars))
    return out.reset_index(drop=True)


def fetch_perp_history(
    *,
    client: Optional[KuCoinRestClient] = None,
    symbol: str = "NEAR-USDTM",
    timeframe: str = "1m",
    lookback_bars: int = 500,
) -> pd.DataFrame:
    c = client or KuCoinRestClient.from_env()
    raw = c.get_futures_candles(symbol=symbol, granularity=_perp_granularity(timeframe))
    rows = []
    for item in raw:
        try:
            ts = _normalize_perp_ts(item[0])
            close = float(item[4])
        except (TypeError, ValueError, IndexError):
            continue
        rows.append({"timestamp": pd.to_datetime(ts, unit="s", utc=True), "perp": close})

    out = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    if lookback_bars > 0:
        out = out.tail(int(lookback_bars))
    return out.reset_index(drop=True)


def align_spot_perp(spot_df: pd.DataFrame, perp_df: pd.DataFrame) -> pd.DataFrame:
    required_spot = {"timestamp", "spot"}
    required_perp = {"timestamp", "perp"}
    if not required_spot.issubset(spot_df.columns):
        raise ValueError(f"spot_df must contain columns {sorted(required_spot)}")
    if not required_perp.issubset(perp_df.columns):
        raise ValueError(f"perp_df must contain columns {sorted(required_perp)}")

    merged = pd.merge(spot_df, perp_df, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged[["timestamp", "spot", "perp"]]


def prepare_market_dataframe(
    *,
    client: Optional[KuCoinRestClient] = None,
    symbol_spot: str = "NEAR-USDT",
    symbol_perp: str = "NEAR-USDTM",
    timeframe: str = "1m",
    lookback_bars: int = 500,
    add_features: bool = False,
    rolling_window: int = 100,
) -> pd.DataFrame:
    c = client or KuCoinRestClient.from_env()
    spot_df = fetch_spot_history(
        client=c,
        symbol=symbol_spot,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
    )
    perp_df = fetch_perp_history(
        client=c,
        symbol=symbol_perp,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
    )
    market_df = align_spot_perp(spot_df=spot_df, perp_df=perp_df)
    if add_features:
        market_df = add_basis_features(market_df, window=rolling_window)
    return market_df

