from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import client_partitions, evaluate_predictions


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | list[int] = 32, hidden_layers: int = 2, activation: str = "ReLU"):
        super().__init__()
        activations: dict[str, type[nn.Module]] = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
        activation_cls = activations.get(activation, nn.ReLU)
        layers: list[nn.Module] = []
        current_dim = input_dim
        hidden_dims = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * max(1, hidden_layers)
        for units in hidden_dims:
            layers.extend([nn.Linear(current_dim, int(units)), activation_cls()])
            current_dim = int(units)
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


@dataclass
class TrainConfig:
    mode: str = "centralized"
    epochs: int = 3
    rounds: int = 3
    clients: int = 4
    local_epochs: int = 1
    batch_size: int = 16
    lr: float = 0.01
    lr_schedule: str = "constant"
    lr_decay: float = 0.5
    lr_step_size: int = 10
    lr_min: float = 1e-4
    hidden_layers: int = 2
    hidden_units: int | list[int] = 32
    activation: str = "ReLU"
    client_fraction: float = 1.0
    dirichlet_alpha: float = 0.3
    clip_norm: float = 1.0
    noise_multiplier: float = 0.2
    epsilon: float = 4.0
    delta: float = 1e-5
    non_iid: bool = False
    seed: int = 42


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total = 0
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
        total += len(features)
    return total_loss / max(total, 1)


def _evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            logits = model(features)
            loss = loss_fn(logits, target)
            total_loss += loss.item() * len(features)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == target).sum().item())
            total += len(features)
    return {"loss": float(total_loss / max(total, 1)), "accuracy": float(correct / max(total, 1))}


def _predict_prob(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(tensor)
        return torch.sigmoid(logits).cpu().numpy()


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _set_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def _average_states(states: list[dict[str, torch.Tensor]], weights: list[int] | None = None) -> dict[str, torch.Tensor]:
    if weights is None:
        normalized = torch.ones(len(states), dtype=torch.float32) / len(states)
    else:
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        normalized = weight_tensor / weight_tensor.sum()
    averaged = {}
    for key in states[0]:
        stacked = torch.stack([state[key] for state in states])
        view_shape = [len(states)] + [1] * (stacked.dim() - 1)
        averaged[key] = (stacked * normalized.reshape(view_shape)).sum(dim=0)
    return averaged


def _make_model(input_dim: int, config: TrainConfig, device: torch.device) -> MLP:
    return MLP(
        input_dim,
        hidden_dim=config.hidden_units,
        hidden_layers=config.hidden_layers,
        activation=config.activation,
    ).to(device)


def _scheduled_lr(config: TrainConfig, step_index: int, total_steps: int | None = None) -> float:
    if config.lr_schedule == "constant":
        return float(config.lr)
    if config.lr_schedule == "step_decay":
        decay_steps = max(1, int(config.lr_step_size))
        return float(max(config.lr_min, config.lr * (config.lr_decay ** (step_index // decay_steps))))
    if config.lr_schedule == "linear_decay":
        if total_steps is None or total_steps <= 1:
            return float(max(config.lr_min, config.lr))
        progress = min(max(step_index, 0), total_steps - 1) / (total_steps - 1)
        return float(max(config.lr_min, config.lr + (config.lr_min - config.lr) * progress))
    return float(config.lr)


def parse_hidden_units(value: Any, hidden_layers: int = 2) -> int | list[int]:
    if isinstance(value, list):
        parsed = [int(item) for item in value if int(item) > 0]
    elif isinstance(value, str) and "," in value:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        return int(value)
    return parsed or [32] * max(1, hidden_layers)


def _client_update(
    base_state: dict[str, torch.Tensor],
    model: nn.Module,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    effective_lr: float | None = None,
) -> dict[str, torch.Tensor]:
    _set_state(model, base_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr if effective_lr is None else effective_lr)
    for _ in range(config.local_epochs):
        _train_epoch(model, loader, optimizer, device)
    updated = _model_state(model)
    if config.mode == "fedavg":
        return updated

    delta = {key: updated[key] - base_state[key] for key in base_state}
    flat = torch.cat([value.flatten() for value in delta.values()])
    norm = torch.linalg.norm(flat)
    scale = min(1.0, config.clip_norm / (norm + 1e-12))
    for key in delta:
        delta[key] = delta[key] * scale
    for key in delta:
        delta[key] = delta[key] + torch.normal(0.0, config.noise_multiplier * config.clip_norm, size=delta[key].shape)
    return {key: base_state[key] + delta[key] for key in base_state}


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainConfig,
    client_partitions_override: list[np.ndarray] | None = None,
    client_labels: list[str] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cpu")
    rng = np.random.default_rng(config.seed)
    model = _make_model(x_train.shape[1], config, device)
    history = {"loss": [], "accuracy": [], "lr": []}
    num_rounds = config.epochs if config.mode == "centralized" else config.rounds

    if config.mode == "centralized":
        loader = _loader(x_train, y_train, config.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        for step_index in range(num_rounds):
            effective_lr = _scheduled_lr(config, step_index, num_rounds)
            for group in optimizer.param_groups:
                group["lr"] = effective_lr
            loss = _train_epoch(model, loader, optimizer, device)
            history["loss"].append(float(loss))
            history["lr"].append(float(effective_lr))
            evaluation = _evaluate_loader(model, loader, device)
            history["accuracy"].append(float(evaluation["accuracy"]))
    else:
        partitions = client_partitions_override or client_partitions(
            y_train,
            num_clients=config.clients,
            non_iid=config.non_iid,
            seed=config.seed,
            alpha=config.dirichlet_alpha,
        )
        for step_index in range(num_rounds):
            effective_lr = _scheduled_lr(config, step_index, num_rounds)
            base_state = _model_state(model)
            client_states = []
            client_weights = []
            selected_count = max(1, int(np.ceil(len(partitions) * config.client_fraction)))
            selected_indices = rng.choice(len(partitions), size=min(selected_count, len(partitions)), replace=False)
            for partition_index in selected_indices:
                partition = partitions[int(partition_index)]
                if len(partition) == 0:
                    continue
                client_loader = _loader(x_train[partition], y_train[partition], config.batch_size, shuffle=True)
                client_model = _make_model(x_train.shape[1], config, device)
                updated_state = _client_update(base_state, client_model, client_loader, config, device, effective_lr)
                client_states.append(updated_state)
                client_weights.append(len(partition))
            if client_states:
                _set_state(model, _average_states(client_states, client_weights))
            evaluation = _evaluate_loader(model, _loader(x_train, y_train, config.batch_size, shuffle=False), device)
            history["loss"].append(float(evaluation["loss"]))
            history["lr"].append(float(effective_lr))
            history["accuracy"].append(float(evaluation["accuracy"]))

    y_prob = _predict_prob(model, x_test, device)
    metrics = evaluate_predictions(y_test, y_prob)
    client_distribution = []
    if config.mode != "centralized":
        partitions = client_partitions_override or client_partitions(y_train, config.clients, config.non_iid, config.seed, config.dirichlet_alpha)
        client_distribution = [
            {
                "client": client_labels[index] if client_labels and index < len(client_labels) else f"client-{index + 1}",
                "size": int(len(partition)),
                "positive": int(y_train[partition].sum()),
                "negative": int(len(partition) - y_train[partition].sum()),
            }
            for index, partition in enumerate(partitions)
        ]
    protocol = (
        "集中式训练：数据合并在同一训练集上训练 MLP，不涉及客户端、参数传输或服务端聚合。"
        if config.mode == "centralized"
        else "服务端托管 FL 模拟：管理端上传并预处理一次数据，服务端按 client_id 列或确定性 4 路切分模拟客户端分区；每个分区训练本地模型更新，FedAvg 按样本量聚合模型参数更新。"
    )
    return {
        "mode": config.mode,
        "metrics": metrics,
        "history": history,
        "predictions": y_prob.tolist(),
        "client_distribution": client_distribution,
        "protocol": protocol,
        "dp": {
            "epsilon": config.epsilon,
            "delta": config.delta,
            "clip_norm": config.clip_norm,
            "noise_multiplier": config.noise_multiplier,
        }
        if config.mode == "dp_fedavg"
        else None,
    }
