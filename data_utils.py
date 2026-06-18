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


ADULT_TARGET_CANDIDATES = ("income", "target", "class")
ADULT_POSITIVE_LABELS = {">50k", ">50k.", "1", "true", "yes"}
ADULT_NEGATIVE_LABELS = {"<=50k", "<=50k.", "0", "false", "no"}


def normalize_missing_markers(frame: pd.DataFrame) -> pd.DataFrame:
    processed = frame.copy()
    return processed.replace({"?": np.nan, " ?": np.nan, "? ": np.nan, "": np.nan})


def infer_adult_target_column(frame: pd.DataFrame, fallback: str = "target") -> str:
    for column in ADULT_TARGET_CANDIDATES:
        if column in frame.columns:
            return column
    return fallback if fallback in frame.columns else (frame.columns[-1] if len(frame.columns) else fallback)


def normalize_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(lambda value: 1 if value in ADULT_POSITIVE_LABELS else 0 if value in ADULT_NEGATIVE_LABELS else np.nan)
    if mapped.notna().all():
        return mapped.astype(int)
    return pd.Series(LabelEncoder().fit_transform(series.astype(str)), index=series.index)


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series.dropna(), errors="coerce")
    return not converted.empty and float(converted.notna().mean()) >= 0.9


def numeric_like_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


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
    target_column = infer_adult_target_column(frame, target_column)
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


def recommended_missing_strategy(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return "mode"
    series = frame[column]
    missing_rate = float(series.isna().mean())
    if missing_rate > 0.4:
        return "drop"
    return "median" if is_numeric_like(series) else "mode"


def recommended_scaler(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or not is_numeric_like(frame[column]):
        return "none"
    series = numeric_like_series(frame[column]).dropna()
    if series.empty or series.nunique() <= 2:
        return "none"
    return "minmax" if abs(float(series.skew())) > 1.0 else "standard"


def apply_column_preprocessing(
    frame: pd.DataFrame,
    target_column: str,
    missing_strategies: dict[str, str] | None = None,
    scaler_strategies: dict[str, str] | None = None,
) -> pd.DataFrame:
    processed = frame.copy()
    missing_strategies = missing_strategies or {}
    scaler_strategies = scaler_strategies or {}

    for column in processed.columns:
        if is_numeric_like(processed[column]):
            converted = numeric_like_series(processed[column])
            if converted.notna().any():
                processed[column] = converted

    drop_columns = [column for column, strategy in missing_strategies.items() if strategy == "drop" and column in processed.columns]
    if drop_columns:
        processed = processed.dropna(subset=drop_columns).reset_index(drop=True)

    for column, strategy in missing_strategies.items():
        if column not in processed.columns or strategy == "drop" or processed[column].isna().sum() == 0:
            continue
        if strategy == "mean" and is_numeric_like(processed[column]):
            value = numeric_like_series(processed[column]).mean()
        elif strategy == "median" and is_numeric_like(processed[column]):
            value = numeric_like_series(processed[column]).median()
        else:
            mode = processed[column].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else 0
        processed[column] = processed[column].fillna(value)

    processed = preprocess_tabular_data(processed, target_column=target_column, missing_strategy="mode", scaler="none")

    for column, strategy in scaler_strategies.items():
        if strategy == "none" or column not in processed.columns or not pd.api.types.is_numeric_dtype(processed[column]):
            continue
        scaler_obj = StandardScaler() if strategy == "standard" else MinMaxScaler()
        processed[[column]] = scaler_obj.fit_transform(processed[[column]])
    return processed


def preprocessing_recommendations(frame: pd.DataFrame, target_column: str = "target") -> dict[str, Any]:
    frame = normalize_missing_markers(frame)
    target_column = infer_adult_target_column(frame, target_column)
    missing = {}
    scalers = {}
    for column in frame.columns:
        if frame[column].isna().any():
            missing[column] = recommended_missing_strategy(frame, column)
        if column not in {target_column, "client_id"} and is_numeric_like(frame[column]):
            scalers[column] = recommended_scaler(frame, column)
    return {"missing_strategies": missing, "scaler_strategies": scalers}


def preprocess_tabular_data(
    frame: pd.DataFrame,
    target_column: str = "target",
    missing_strategy: str = "drop",
    scaler: str = "standard",
) -> pd.DataFrame:
    if missing_strategy not in {"drop", "mean", "median", "mode"}:
        raise ValueError("missing_strategy must be one of: drop, mean, median, mode")
    if scaler not in {"none", "standard", "minmax"}:
        raise ValueError("scaler must be one of: none, standard, minmax")

    processed = normalize_missing_markers(frame)
    target_column = infer_adult_target_column(processed, target_column)
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

    if target_column in processed.columns:
        processed[target_column] = normalize_binary_target(processed[target_column])

    categorical_features = [
        column
        for column in feature_columns
        if column in processed.columns and not pd.api.types.is_numeric_dtype(processed[column])
    ]
    if categorical_features:
        processed = pd.get_dummies(processed, columns=categorical_features, prefix=categorical_features, dummy_na=False)

    feature_columns = [column for column in processed.columns if column not in {target_column, "client_id"}]
    numeric_features = [column for column in feature_columns if column in processed.columns and pd.api.types.is_numeric_dtype(processed[column])]
    if numeric_features and scaler != "none":
        scaler_obj = StandardScaler() if scaler == "standard" else MinMaxScaler()
        processed[numeric_features] = scaler_obj.fit_transform(processed[numeric_features])
    return processed


def split_features_target(frame: pd.DataFrame, target_column: str = "target") -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_column = infer_adult_target_column(frame, target_column)
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
    num_clients = max(1, int(num_clients))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    if not non_iid:
        rng.shuffle(indices)
        return [np.asarray(partition, dtype=int) for partition in np.array_split(indices, num_clients)]

    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    concentration = max(float(alpha), 1e-6)
    for label in np.unique(y):
        class_indices = indices[y == label]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(concentration, num_clients))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for client_index, client_indices in enumerate(np.split(class_indices, split_points)):
            partitions[client_index].extend(client_indices.tolist())
    result = []
    for partition in partitions:
        shuffled = np.array(partition, dtype=int)
        rng.shuffle(shuffled)
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
