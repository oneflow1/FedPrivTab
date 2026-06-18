from __future__ import annotations

import numpy as np
import pandas as pd

from data_utils import client_partitions, generate_sample_data, preprocess_tabular_data, validate_tabular_data
from training import MLP, TrainConfig, parse_hidden_units, train_model


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
    assert any(column.startswith("feature_b_") for column in processed.columns)
    encoded_columns = [column for column in processed.columns if column.startswith("feature_b_")]
    assert all(processed[column].dtype.kind in {"b", "i", "u", "f"} for column in encoded_columns)
    assert set(processed["target"].unique()) == {0, 1}
    assert validate_tabular_data(pd.concat([processed] * 5, ignore_index=True), target_column="target").valid


def test_dirichlet_partitions_preserve_all_samples() -> None:
    y = np.array([0, 1] * 30)
    partitions = client_partitions(y, num_clients=4, non_iid=True, seed=9, alpha=0.2)
    merged = np.sort(np.concatenate(partitions))
    assert merged.tolist() == list(range(len(y)))
    assert len(partitions) == 4


def test_dirichlet_alpha_controls_label_skew() -> None:
    y = np.array([0] * 300 + [1] * 300)
    low_alpha = client_partitions(y, num_clients=8, non_iid=True, seed=17, alpha=0.05)
    high_alpha = client_partitions(y, num_clients=8, non_iid=True, seed=17, alpha=10.0)

    def mean_label_skew(partitions: list[np.ndarray]) -> float:
        skews = []
        for partition in partitions:
            if len(partition) == 0:
                continue
            positive_rate = float(y[partition].mean())
            skews.append(abs(positive_rate - 0.5))
        return float(np.mean(skews))

    assert mean_label_skew(low_alpha) > mean_label_skew(high_alpha) + 0.15


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
    assert "accuracy" in fedavg["history"]
    assert len(centralized["history"]["accuracy"]) == 1
    assert centralized["history"]["lr"] == [0.01]
    assert fedavg["history"]["lr"] == [0.01]
    assert len(dp_fedavg["history"]["accuracy"]) == 1
    assert dp_fedavg["dp"] is not None
    assert {"epsilon", "delta", "clip_norm", "noise_multiplier"} <= set(dp_fedavg["dp"])


def test_learning_rate_history_supports_step_decay_for_centralized_and_fedavg() -> None:
    frame = generate_sample_data(samples=96, features=4, clients=3, seed=21)
    x = frame.drop(columns=["target", "client_id"]).to_numpy()
    y = frame["target"].to_numpy()
    split = 72
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    centralized = train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        TrainConfig(
            mode="centralized",
            epochs=5,
            lr=0.08,
            lr_schedule="step_decay",
            lr_decay=0.5,
            lr_step_size=2,
            lr_min=0.02,
            hidden_layers=1,
            hidden_units=8,
            seed=21,
        ),
    )
    assert centralized["history"]["lr"] == [0.08, 0.08, 0.04, 0.04, 0.02]

    partitions = [np.arange(client_index, len(y_train), 3) for client_index in range(3)]
    fedavg = train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        TrainConfig(
            mode="fedavg",
            rounds=5,
            clients=3,
            local_epochs=1,
            lr=0.08,
            lr_schedule="step_decay",
            lr_decay=0.5,
            lr_step_size=2,
            lr_min=0.02,
            hidden_layers=1,
            hidden_units=8,
            seed=21,
        ),
        client_partitions_override=partitions,
    )
    assert fedavg["history"]["lr"] == [0.08, 0.08, 0.04, 0.04, 0.02]
    assert len(fedavg["history"]["lr"]) == len(fedavg["history"]["accuracy"]) == 5


def test_fedavg_learns_without_dp_clipping() -> None:
    rng = np.random.default_rng(123)
    negatives = rng.normal(loc=(-2.0, -2.0), scale=0.35, size=(80, 2))
    positives = rng.normal(loc=(2.0, 2.0), scale=0.35, size=(80, 2))
    x = np.vstack([negatives, positives]).astype(np.float32)
    y = np.array([0] * 80 + [1] * 80)
    permutation = rng.permutation(len(y))
    x = x[permutation]
    y = y[permutation]
    split = 120
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    partitions = [np.arange(client_index, len(y_train), 4) for client_index in range(4)]

    result = train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        TrainConfig(
            mode="fedavg",
            rounds=5,
            clients=4,
            local_epochs=2,
            batch_size=16,
            lr=0.05,
            hidden_layers=1,
            hidden_units=8,
            activation="Tanh",
            clip_norm=1e-6,
            seed=7,
        ),
        client_partitions_override=partitions,
    )

    assert result["history"]["accuracy"][-1] > result["history"]["accuracy"][0]
    assert result["metrics"]["f1"] > 0.9
    assert (np.array(result["predictions"]) >= 0.5).sum() > 0


def test_hidden_layer_structure_accepts_requirement_default() -> None:
    parsed = parse_hidden_units("64,32", hidden_layers=2)
    assert parsed == [64, 32]
    model = MLP(input_dim=4, hidden_dim=parsed, hidden_layers=2)
    linear_layers = [layer for layer in model.net if layer.__class__.__name__ == "Linear"]
    assert linear_layers[0].out_features == 64
    assert linear_layers[1].out_features == 32
