from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .config import AppConfig
from .forecast_strategy import PRICE_MODEL_FEATURE_COLUMNS, split_feature_frame


@dataclass
class CatBoostClassifierArtifacts:
    feature_columns: list[str]
    classifier: CatBoostClassifier
    label_threshold: float
    metrics: dict[str, float]


def _label_from_change(value: float, threshold: float) -> int:
    if value > threshold:
        return 2
    if value < -threshold:
        return 0
    return 1


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores: list[float] = []
    for cls in (0, 1, 2):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = 0.0 if tp + fp == 0.0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0.0 else tp / (tp + fn)
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def train_catboost_classifier(feature_frame: pd.DataFrame, cfg: AppConfig) -> CatBoostClassifierArtifacts:
    split = split_feature_frame(feature_frame, cfg)
    feature_columns = list(PRICE_MODEL_FEATURE_COLUMNS)

    train_frame = split.train_feature_frame.copy()
    test_frame = split.test_feature_frame.copy()
    train_frame["target_basis_change"] = train_frame["basis"].shift(-1) - train_frame["basis"]
    test_frame["target_basis_change"] = test_frame["basis"].shift(-1) - test_frame["basis"]
    train_frame = train_frame.dropna().reset_index(drop=True)
    test_frame = test_frame.dropna().reset_index(drop=True)

    label_threshold = float(np.quantile(np.abs(train_frame["target_basis_change"].to_numpy(dtype=float)), 0.67))
    y_train = train_frame["target_basis_change"].map(lambda v: _label_from_change(float(v), label_threshold))
    y_test = test_frame["target_basis_change"].map(lambda v: _label_from_change(float(v), label_threshold))

    classifier = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.03,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )
    classifier.fit(train_frame[feature_columns], y_train)

    pred = np.asarray(classifier.predict(test_frame[feature_columns]), dtype=int).reshape(-1)
    actual = y_test.to_numpy(dtype=int)
    accuracy = float(np.mean(pred == actual))
    macro_f1 = _macro_f1_score(actual, pred)

    return CatBoostClassifierArtifacts(
        feature_columns=feature_columns,
        classifier=classifier,
        label_threshold=label_threshold,
        metrics={
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },
    )
