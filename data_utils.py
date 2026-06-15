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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


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


def validate_tabular_data(
    frame: pd.DataFrame,
    target_column: str = "target",
    min_samples: int = 20,
    max_missing_rate: float = 0.2,
) -> ValidationResult:
    if frame.empty:
        return ValidationResult(False, "数据为空", {"rows": 0})
    if target_column not in frame.columns:
        return ValidationResult(False, f"缺少目标列: {target_column}", {"columns": list(frame.columns)})
    if len(frame) < min_samples:
        return ValidationResult(False, f"样本量不足，至少需要 {min_samples} 行", {"rows": len(frame)})
    missing = int(frame.isna().sum().sum())
    missing_rate = float(missing / max(frame.size, 1))
    if missing_rate > max_missing_rate:
        return ValidationResult(False, "缺失率超过阈值，请先清洗", {"missing_values": missing, "missing_rate": missing_rate})
    if missing:
        return ValidationResult(False, "存在缺失值，请先清洗", {"missing_values": missing, "missing_rate": missing_rate})
    labels = frame[target_column].dropna().unique()
    if len(labels) != 2:
        return ValidationResult(False, "当前 MLP 示例仅支持二分类标签", {"label_classes": [str(label) for label in labels]})
    feature_columns = [column for column in frame.columns if column not in {target_column, "client_id"}]
    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        return ValidationResult(False, "存在非数值特征，请先执行类别编码", {"non_numeric_features": non_numeric})
    return ValidationResult(
        True,
        "数据校验通过",
        {
            "rows": len(frame),
            "features": len(feature_columns),
            "columns": list(frame.columns),
            "missing_rate": missing_rate,
            "label_classes": [str(label) for label in sorted(labels)],
        },
    )


def preprocess_tabular_data(
    frame: pd.DataFrame,
    target_column: str = "target",
    missing_strategy: str = "drop",
    scaler: str = "standard",
) -> pd.DataFrame:
    processed = frame.copy()
    feature_columns = [column for column in processed.columns if column not in {target_column, "client_id"}]

    if missing_strategy == "drop":
        processed = processed.dropna().reset_index(drop=True)
    else:
        for column in processed.columns:
            if processed[column].isna().sum() == 0:
                continue
            if missing_strategy == "mean" and pd.api.types.is_numeric_dtype(processed[column]):
                value = processed[column].mean()
            elif missing_strategy == "median" and pd.api.types.is_numeric_dtype(processed[column]):
                value = processed[column].median()
            else:
                mode = processed[column].mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else 0
            processed[column] = processed[column].fillna(value)

    if target_column in processed.columns and not pd.api.types.is_numeric_dtype(processed[target_column]):
        processed[target_column] = LabelEncoder().fit_transform(processed[target_column].astype(str))

    for column in feature_columns:
        if column in processed.columns and not pd.api.types.is_numeric_dtype(processed[column]):
            processed[column] = LabelEncoder().fit_transform(processed[column].astype(str))

    numeric_features = [column for column in feature_columns if column in processed.columns and pd.api.types.is_numeric_dtype(processed[column])]
    if numeric_features and scaler != "none":
        scaler_obj = StandardScaler() if scaler == "standard" else MinMaxScaler()
        processed[numeric_features] = scaler_obj.fit_transform(processed[numeric_features])
    return processed


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


def client_partitions(
    y: np.ndarray,
    num_clients: int = 4,
    non_iid: bool = False,
    seed: int = 42,
    alpha: float = 0.3,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    if not non_iid:
        rng.shuffle(indices)
        return [partition for partition in np.array_split(indices, num_clients) if len(partition)]

    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(y):
        class_indices = indices[y == label]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(max(alpha, 1e-3), num_clients))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for client_index, client_indices in enumerate(np.split(class_indices, split_points)):
            partitions[client_index].extend(client_indices.tolist())
    result = []
    for partition in partitions:
        shuffled = np.array(partition, dtype=int)
        rng.shuffle(shuffled)
        if len(shuffled):
            result.append(shuffled)
    return result


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
