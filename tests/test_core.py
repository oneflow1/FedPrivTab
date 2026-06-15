from __future__ import annotations

import numpy as np
import pandas as pd

from data_utils import client_partitions, generate_sample_data, preprocess_tabular_data, validate_tabular_data
from training import TrainConfig, train_model


def test_generate_and_validate_sample_data() -> None:
    frame = generate_sample_data(samples=60, features=5, clients=3, seed=7)
    result = validate_tabular_data(frame)
    assert result.valid
    assert "target" in frame.columns


def test_preprocess_encodes_fills_and_scales() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1.0, None, 3.0, 4.0],
            "feature_b": ["x", "y", "x", "z"],
            "target": ["no", "yes", "no", "yes"],
            "client_id": [0, 0, 1, 1],
        }
    )
    processed = preprocess_tabular_data(frame, target_column="target", missing_strategy="mean", scaler="minmax")
    assert not processed.isna().any().any()
    assert processed["feature_b"].dtype.kind in {"i", "u", "f"}
    assert set(processed["target"].unique()) == {0, 1}
    assert validate_tabular_data(pd.concat([processed] * 5, ignore_index=True), target_column="target").valid


def test_dirichlet_partitions_preserve_all_samples() -> None:
    y = np.array([0, 1] * 30)
    partitions = client_partitions(y, num_clients=4, non_iid=True, seed=9, alpha=0.2)
    merged = np.sort(np.concatenate(partitions))
    assert merged.tolist() == list(range(len(y)))
    assert len(partitions) <= 4


def test_training_modes_run() -> None:
    frame = generate_sample_data(samples=80, features=4, clients=4, seed=11)
    feature_frame = frame.drop(columns=[])
    x = feature_frame.drop(columns=["target", "client_id"]).to_numpy()
    y = feature_frame["target"].to_numpy()
    split = 60
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    centralized = train_model(x_train, y_train, x_test, y_test, TrainConfig(mode="centralized", epochs=1, seed=11, hidden_layers=1, activation="Tanh"))
    fedavg = train_model(x_train, y_train, x_test, y_test, TrainConfig(mode="fedavg", rounds=1, clients=2, seed=11, client_fraction=0.5))
    dp_fedavg = train_model(x_train, y_train, x_test, y_test, TrainConfig(mode="dp_fedavg", rounds=1, clients=2, seed=11, dirichlet_alpha=0.4))
    assert "accuracy" in centralized["metrics"]
    assert "loss" in fedavg["history"]
    assert dp_fedavg["dp"] is not None
