from __future__ import annotations

import pandas as pd

from data_utils import generate_sample_data, validate_tabular_data
from training import TrainConfig, train_model


def test_generate_and_validate_sample_data() -> None:
    frame = generate_sample_data(samples=60, features=5, clients=3, seed=7)
    result = validate_tabular_data(frame)
    assert result.valid
    assert "target" in frame.columns


def test_training_modes_run() -> None:
    frame = generate_sample_data(samples=80, features=4, clients=4, seed=11)
    feature_frame = frame.drop(columns=[])
    x = feature_frame.drop(columns=["target", "client_id"]).to_numpy()
    y = feature_frame["target"].to_numpy()
    split = 60
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    centralized = train_model(x_train, y_train, x_test, y_test, TrainConfig(mode="centralized", epochs=1, seed=11))
    fedavg = train_model(x_train, y_train, x_test, y_test, TrainConfig(mode="fedavg", rounds=1, clients=2, seed=11))
    dp_fedavg = train_model(x_train, y_train, x_test, y_test, TrainConfig(mode="dp_fedavg", rounds=1, clients=2, seed=11))
    assert "accuracy" in centralized["metrics"]
    assert "loss" in fedavg["history"]
    assert dp_fedavg["dp"] is not None
