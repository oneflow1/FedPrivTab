from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class ValidationResult:
    valid: bool
    message: str
    details: dict[str, Any]


def generate_sample_data(samples: int = 240, features: int = 6, clients: int = 4, seed: int = 42) -> pd.DataFrame:
    x, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=max(2, features - 2),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.6, 0.4],
        class_sep=1.2,
        random_state=seed,
    )
    frame = pd.DataFrame(x, columns=[f"feature_{index}" for index in range(features)])
    frame["target"] = y
    frame["client_id"] = np.arange(samples) % max(1, clients)
    return frame


def validate_tabular_data(frame: pd.DataFrame, target_column: str = "target", min_samples: int = 20) -> ValidationResult:
    if frame.empty:
        return ValidationResult(False, "数据为空", {"rows": 0})
    if target_column not in frame.columns:
        return ValidationResult(False, f"缺少目标列: {target_column}", {"columns": list(frame.columns)})
    if len(frame) < min_samples:
        return ValidationResult(False, f"样本量不足，至少需要 {min_samples} 行", {"rows": len(frame)})
    missing = int(frame.isna().sum().sum())
    if missing:
        return ValidationResult(False, "存在缺失值，请先清洗", {"missing_values": missing})
    return ValidationResult(True, "数据校验通过", {"rows": len(frame), "columns": list(frame.columns)})


def split_features_target(frame: pd.DataFrame, target_column: str = "target") -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_columns = [column for column in frame.columns if column != target_column and column != "client_id"]
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame[target_column].to_numpy(dtype=np.int64)
    return x, y, feature_columns


def train_test_data(frame: pd.DataFrame, target_column: str = "target", test_size: float = 0.25, seed: int = 42):
    x, y, feature_columns = split_features_target(frame, target_column)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test, feature_columns


def client_partitions(y: np.ndarray, num_clients: int = 4, non_iid: bool = False, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    if not non_iid:
        rng.shuffle(indices)
        return [partition for partition in np.array_split(indices, num_clients) if len(partition)]
    class_0 = indices[y == 0]
    class_1 = indices[y == 1]
    rng.shuffle(class_0)
    rng.shuffle(class_1)
    partitions = []
    for client_index in range(num_clients):
        if client_index % 2 == 0:
            take = class_0[client_index::num_clients]
            if take.size == 0:
                take = class_1[client_index::num_clients]
        else:
            take = class_1[client_index::num_clients]
            if take.size == 0:
                take = class_0[client_index::num_clients]
        partitions.append(np.array(take, dtype=int))
    return [partition for partition in partitions if len(partition)]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = None
    return metrics
