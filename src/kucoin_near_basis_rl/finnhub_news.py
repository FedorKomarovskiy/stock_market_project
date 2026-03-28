from __future__ import annotations

import base64
import os
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import NewsConfig


NEWS_FEATURE_COLUMNS = [
    "news_count_6h",
    "news_sentiment_mean_6h",
    "news_weighted_sentiment_mean_6h",
    "news_sentiment_sum_6h",
    "news_sentiment_abs_sum_6h",
    "news_relevance_mean_6h",
    "news_near_count_6h",
    "news_positive_count_6h",
    "news_negative_count_6h",
    "news_cryptopanic_count_6h",
    "news_finnhub_count_6h",
    "news_x_count_6h",
    "news_source_diversity_6h",
    "news_count_24h",
    "news_sentiment_mean_24h",
    "news_weighted_sentiment_mean_24h",
    "news_sentiment_sum_24h",
    "news_sentiment_abs_sum_24h",
    "news_relevance_mean_24h",
    "news_near_count_24h",
    "news_positive_count_24h",
    "news_negative_count_24h",
    "news_cryptopanic_count_24h",
    "news_finnhub_count_24h",
    "news_x_count_24h",
    "news_source_diversity_24h",
    "news_burst_ratio_6h_24h",
]

POSITIVE_WORDS = {
    "bullish": 1.4,
    "surge": 1.4,
    "rally": 1.3,
    "gain": 1.0,
    "gains": 1.0,
    "growth": 1.1,
    "beat": 1.1,
    "beats": 1.1,
    "strong": 0.8,
    "upgrade": 1.0,
    "adoption": 1.3,
    "partnership": 1.1,
    "launch": 0.7,
    "record": 0.8,
    "breakout": 1.4,
    "optimism": 1.2,
    "positive": 0.8,
    "recovery": 0.8,
    "inflow": 1.2,
    "approval": 1.0,
    "expand": 0.7,
    "expansion": 0.7,
    "momentum": 0.8,
    "buy": 0.7,
    "accumulate": 1.0,
    "rebounds": 1.0,
    "rebound": 1.0,
    "outperform": 1.2,
    "listing": 0.8,
}
NEGATIVE_WORDS = {
    "bearish": 1.4,
    "drop": 1.1,
    "drops": 1.1,
    "fall": 1.0,
    "falls": 1.0,
    "crash": 1.6,
    "selloff": 1.4,
    "loss": 1.0,
    "losses": 1.0,
    "weak": 0.8,
    "downgrade": 1.1,
    "hack": 1.7,
    "lawsuit": 1.2,
    "ban": 1.2,
    "decline": 1.0,
    "declines": 1.0,
    "risk": 0.7,
    "negative": 0.8,
    "outflow": 1.1,
    "volatility": 0.4,
    "fraud": 1.8,
    "default": 1.6,
    "liquidation": 1.4,
    "fear": 1.0,
    "sell": 0.7,
    "dump": 1.4,
    "exploit": 1.8,
    "breach": 1.7,
    "delist": 1.5,
    "investigation": 1.1,
}
NEAR_PATTERNS = [
    re.compile(r"\bnear protocol\b", re.IGNORECASE),
    re.compile(r"\bnear foundation\b", re.IGNORECASE),
    re.compile(r"\bnear-usdt\b", re.IGNORECASE),
    re.compile(r"\bnearusdt\b", re.IGNORECASE),
    re.compile(r"\bnear\b", re.IGNORECASE),
]


@dataclass
class NewsRow:
    id: str
    timestamp: datetime
    headline: str
    summary: str
    source: str
    source_type: str
    sentiment_score: float
    relevance_score: float
    near_mentions: int


class MultiSourceNewsClient:
    def __init__(self, cfg: NewsConfig) -> None:
        self.cfg = cfg

    def fetch_all_news(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if self.cfg.use_finnhub:
            rows.extend(self._fetch_finnhub_rows())
        if self.cfg.use_cryptopanic:
            rows.extend(self._fetch_cryptopanic_rows())
        if self.cfg.use_x:
            rows.extend(self._fetch_x_rows())
        if not rows:
            return pd.DataFrame(columns=_empty_columns())
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values("timestamp").drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

    def _fetch_finnhub_rows(self) -> list[dict[str, Any]]:
        api_key = os.getenv(self.cfg.api_key_env, "").strip()
        if not api_key:
            return []
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": self.cfg.category, "token": api_key},
                timeout=self.cfg.request_timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in payload:
            headline = str(item.get("headline", "") or "")
            summary = str(item.get("summary", "") or "")
            text = f"{headline} {summary}".strip()
            rows.append(_build_news_row(
                item_id=f"finnhub-{item.get('id', 0)}",
                timestamp=datetime.fromtimestamp(int(item.get("datetime", 0) or 0), tz=timezone.utc),
                headline=headline,
                summary=summary,
                source=str(item.get("source", "") or "Finnhub"),
                source_type="finnhub",
                text=text,
            ))
        return rows

    def _fetch_cryptopanic_rows(self) -> list[dict[str, Any]]:
        api_key = os.getenv(self.cfg.cryptopanic_api_key_env, "").strip()
        if not api_key:
            return []
        try:
            response = requests.get(
                "https://cryptopanic.com/api/developer/v2/posts/",
                params={"auth_token": api_key, "currencies": "NEAR", "kind": "news", "public": "true"},
                timeout=self.cfg.request_timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        results = payload.get("results", []) if isinstance(payload, dict) else []
        rows: list[dict[str, Any]] = []
        for item in results:
            headline = str(item.get("title", "") or "")
            summary = str(item.get("description", "") or "")
            text = f"{headline} {summary}".strip()
            published_at = str(item.get("published_at", "") or "")
            if not published_at:
                continue
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00")).astimezone(timezone.utc)
            rows.append(_build_news_row(
                item_id=f"cryptopanic-{item.get('id', 0)}",
                timestamp=dt,
                headline=headline,
                summary=summary,
                source=str(item.get("source", {}).get("title", "") if isinstance(item.get("source"), dict) else item.get("source", "") or "CryptoPanic"),
                source_type="cryptopanic",
                text=text,
            ))
        return rows

    def _fetch_x_rows(self) -> list[dict[str, Any]]:
        bearer = os.getenv(self.cfg.x_bearer_env, "").strip()
        if not bearer:
            bearer = _build_x_bearer_from_consumer_keys(
                consumer_key=os.getenv(self.cfg.x_consumer_key_env, "").strip(),
                consumer_secret=os.getenv(self.cfg.x_consumer_secret_env, "").strip(),
                timeout=self.cfg.request_timeout_sec,
            )
        if not bearer:
            return []
        try:
            response = requests.get(
                "https://api.twitter.com/2/tweets/search/recent",
                headers={"Authorization": f"Bearer {urllib.parse.unquote(bearer)}"},
                params={
                    "query": self.cfg.x_query,
                    "max_results": int(self.cfg.x_max_results),
                    "tweet.fields": "created_at,public_metrics,lang,author_id,text",
                },
                timeout=self.cfg.request_timeout_sec,
            )
            if response.status_code in {401, 402, 403}:
                return []
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for item in payload.get("data", []) if isinstance(payload, dict) else []:
            text = str(item.get("text", "") or "")
            created_at = str(item.get("created_at", "") or "")
            if not created_at:
                continue
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(timezone.utc)
            rows.append(_build_news_row(
                item_id=f"x-{item.get('id', '')}",
                timestamp=dt,
                headline=text[:240],
                summary="",
                source="X",
                source_type="x",
                text=text,
            ))
        return rows


def _build_x_bearer_from_consumer_keys(consumer_key: str, consumer_secret: str, timeout: int) -> str:
    if not consumer_key or not consumer_secret:
        return ""
    cred = f"{urllib.parse.quote(consumer_key)}:{urllib.parse.quote(consumer_secret)}".encode()
    headers = {
        "Authorization": "Basic " + base64.b64encode(cred).decode(),
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    }
    try:
        response = requests.post(
            "https://api.twitter.com/oauth2/token",
            headers=headers,
            data={"grant_type": "client_credentials"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return ""
    return str(payload.get("access_token", "") or "")


def _build_news_row(
    item_id: str,
    timestamp: datetime,
    headline: str,
    summary: str,
    source: str,
    source_type: str,
    text: str,
) -> dict[str, Any]:
    sentiment_score = _combined_sentiment_score(headline, summary)
    return {
        "id": str(item_id),
        "timestamp": timestamp,
        "headline": headline,
        "summary": summary,
        "source": source,
        "source_type": source_type,
        "sentiment_score": float(sentiment_score),
        "relevance_score": float(_headline_relevance_score(headline, summary)),
        "near_mentions": int(_near_mentions(text)),
    }


def _empty_columns() -> list[str]:
    return ["id", "timestamp", "headline", "summary", "source", "source_type", "sentiment_score", "relevance_score", "near_mentions"]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())


def _lexicon_sentiment_score(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos = sum(float(POSITIVE_WORDS.get(token, 0.0)) for token in tokens)
    neg = sum(float(NEGATIVE_WORDS.get(token, 0.0)) for token in tokens)
    if pos <= 0.0 and neg <= 0.0:
        return 0.0
    return float((pos - neg) / max(pos + neg, 1.0))


def _combined_sentiment_score(headline: str, summary: str) -> float:
    headline_score = _lexicon_sentiment_score(headline)
    summary_score = _lexicon_sentiment_score(summary)
    blended = (0.7 * headline_score) + (0.3 * summary_score)
    headline_lower = headline.lower()
    if "not " in headline_lower or "no " in headline_lower:
        blended *= 0.85
    if "!" in headline:
        blended *= 1.05
    return float(np.clip(blended, -1.0, 1.0))


def _headline_relevance_score(headline: str, summary: str) -> float:
    text = f"{headline} {summary}".lower()
    relevance = 0.0
    if "crypto" in text or "bitcoin" in text or "ethereum" in text or "near" in text:
        relevance += 0.5
    if any(pattern.search(text) for pattern in NEAR_PATTERNS):
        relevance += 0.5
    return min(relevance, 1.0)


def _near_mentions(text: str) -> int:
    return int(any(pattern.search(text) for pattern in NEAR_PATTERNS))


def _load_cached_news(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=_empty_columns())
    frame = pd.read_csv(cache_path)
    for column in _empty_columns():
        if column not in frame.columns:
            frame[column] = np.nan
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=_empty_columns())
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("UTC")
    return frame


def _save_cached_news(cache_path: Path, frame: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)


def load_news_frame(cfg: NewsConfig) -> pd.DataFrame:
    cache_path = Path(cfg.cache_path)
    cached = _load_cached_news(cache_path)
    try:
        fresh = MultiSourceNewsClient(cfg).fetch_all_news()
    except Exception:
        fresh = pd.DataFrame(columns=_empty_columns())
    frames = [frame for frame in (cached, fresh) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_empty_columns())
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    allowed_sources: set[str] = set()
    if cfg.use_finnhub:
        allowed_sources.add("finnhub")
    if cfg.use_cryptopanic:
        allowed_sources.add("cryptopanic")
    if cfg.use_x:
        allowed_sources.add("x")
    combined["source_type"] = combined["source_type"].astype(str)
    combined = combined[combined["source_type"].isin(allowed_sources)]
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.cache_retention_days)
    combined = combined[combined["timestamp"] >= cutoff]
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    _save_cached_news(cache_path, combined)
    return combined


def build_news_feature_frame(raw_frame: pd.DataFrame, cfg: NewsConfig) -> pd.DataFrame:
    timestamps = pd.to_datetime(raw_frame["timestamp"], utc=True)
    output = pd.DataFrame({"timestamp": timestamps})
    for col in NEWS_FEATURE_COLUMNS:
        output[col] = 0.0

    news_frame = load_news_frame(cfg)
    if news_frame.empty:
        return output

    news_ts = pd.to_datetime(news_frame["timestamp"], utc=True).astype("int64").to_numpy()
    market_ts = timestamps.astype("int64").to_numpy()
    sentiment = news_frame["sentiment_score"].to_numpy(dtype=np.float64)
    sentiment_abs = np.abs(sentiment)
    relevance = news_frame["relevance_score"].to_numpy(dtype=np.float64)
    weighted_sentiment = sentiment * np.maximum(relevance, 0.25)
    near_mentions = news_frame["near_mentions"].to_numpy(dtype=np.float64)
    ones = np.ones(len(news_frame), dtype=np.float64)
    pos_mask = (sentiment > 0.05).astype(np.float64)
    neg_mask = (sentiment < -0.05).astype(np.float64)
    finnhub_mask = (news_frame["source_type"] == "finnhub").to_numpy(dtype=np.float64)
    cp_mask = (news_frame["source_type"] == "cryptopanic").to_numpy(dtype=np.float64)
    x_mask = (news_frame["source_type"] == "x").to_numpy(dtype=np.float64)

    for hours in (int(cfg.short_window_hours), int(cfg.long_window_hours)):
        suffix = f"{hours}h"
        left = np.searchsorted(news_ts, market_ts - int(pd.Timedelta(hours=hours).value), side="right")
        right = np.searchsorted(news_ts, market_ts, side="right")
        counts = _window_sum(ones, left, right)
        sentiment_sum = _window_sum(sentiment, left, right)
        sentiment_abs_sum = _window_sum(sentiment_abs, left, right)
        weighted_sentiment_sum = _window_sum(weighted_sentiment, left, right)
        relevance_sum = _window_sum(relevance, left, right)
        near_sum = _window_sum(near_mentions, left, right)
        pos_sum = _window_sum(pos_mask, left, right)
        neg_sum = _window_sum(neg_mask, left, right)
        finnhub_sum = _window_sum(finnhub_mask, left, right)
        cp_sum = _window_sum(cp_mask, left, right)
        x_sum = _window_sum(x_mask, left, right)
        with np.errstate(divide="ignore", invalid="ignore"):
            sentiment_mean = np.where(counts > 0, sentiment_sum / counts, 0.0)
            weighted_sentiment_mean = np.where(relevance_sum > 0, weighted_sentiment_sum / relevance_sum, sentiment_mean)
            relevance_mean = np.where(counts > 0, relevance_sum / counts, 0.0)
        output[f"news_count_{suffix}"] = counts
        output[f"news_sentiment_mean_{suffix}"] = sentiment_mean
        output[f"news_weighted_sentiment_mean_{suffix}"] = weighted_sentiment_mean
        output[f"news_sentiment_sum_{suffix}"] = sentiment_sum
        output[f"news_sentiment_abs_sum_{suffix}"] = sentiment_abs_sum
        output[f"news_relevance_mean_{suffix}"] = relevance_mean
        output[f"news_near_count_{suffix}"] = near_sum
        output[f"news_positive_count_{suffix}"] = pos_sum
        output[f"news_negative_count_{suffix}"] = neg_sum
        output[f"news_cryptopanic_count_{suffix}"] = cp_sum
        output[f"news_finnhub_count_{suffix}"] = finnhub_sum
        output[f"news_x_count_{suffix}"] = x_sum
        output[f"news_source_diversity_{suffix}"] = (
            (finnhub_sum > 0).astype(float) + (cp_sum > 0).astype(float) + (x_sum > 0).astype(float)
        )
    output["news_burst_ratio_6h_24h"] = np.where(
        output["news_count_24h"].to_numpy(dtype=np.float64) > 0.0,
        output["news_count_6h"].to_numpy(dtype=np.float64) / np.maximum(output["news_count_24h"].to_numpy(dtype=np.float64), 1.0),
        0.0,
    )
    return output


def _window_sum(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    return cumsum[right] - cumsum[left]
