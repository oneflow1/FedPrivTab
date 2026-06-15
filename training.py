from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import client_partitions, evaluate_predictions


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

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
    clip_norm: float = 1.0
    noise_multiplier: float = 0.2
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


def _client_update(base_state: dict[str, torch.Tensor], model: nn.Module, loader: DataLoader, config: TrainConfig, device: torch.device) -> dict[str, torch.Tensor]:
    _set_state(model, base_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    for _ in range(config.local_epochs):
        _train_epoch(model, loader, optimizer, device)
    updated = _model_state(model)
    delta = {key: updated[key] - base_state[key] for key in base_state}
    flat = torch.cat([value.flatten() for value in delta.values()])
    norm = torch.linalg.norm(flat)
    scale = min(1.0, config.clip_norm / (norm + 1e-12))
    for key in delta:
        delta[key] = delta[key] * scale
    if config.mode == "dp_fedavg":
        for key in delta:
            delta[key] = delta[key] + torch.normal(0.0, config.noise_multiplier * config.clip_norm, size=delta[key].shape)
    return {key: base_state[key] + delta[key] for key in base_state}


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, config: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cpu")
    model = MLP(x_train.shape[1]).to(device)
    history = {"loss": []}
    num_rounds = config.epochs if config.mode == "centralized" else config.rounds

    if config.mode == "centralized":
        loader = _loader(x_train, y_train, config.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        for _ in range(num_rounds):
            loss = _train_epoch(model, loader, optimizer, device)
            history["loss"].append(float(loss))
    else:
        partitions = client_partitions(y_train, num_clients=config.clients, non_iid=config.non_iid, seed=config.seed)
        for _ in range(num_rounds):
            base_state = _model_state(model)
            client_states = []
            client_weights = []
            for partition in partitions:
                if len(partition) == 0:
                    continue
                client_loader = _loader(x_train[partition], y_train[partition], config.batch_size, shuffle=True)
                client_model = MLP(x_train.shape[1]).to(device)
                updated_state = _client_update(base_state, client_model, client_loader, config, device)
                client_states.append(updated_state)
                client_weights.append(len(partition))
            if client_states:
                _set_state(model, _average_states(client_states, client_weights))
            eval_loader = _loader(x_train, y_train, config.batch_size, shuffle=False)
            with torch.no_grad():
                loss_fn = nn.BCEWithLogitsLoss()
                total_loss = 0.0
                total = 0
                for features, target in eval_loader:
                    logits = model(features.to(device))
                    loss = loss_fn(logits, target.to(device))
                    total_loss += loss.item() * len(features)
                    total += len(features)
            history["loss"].append(float(total_loss / max(total, 1)))

    y_prob = _predict_prob(model, x_test, device)
    metrics = evaluate_predictions(y_test, y_prob)
    return {
        "mode": config.mode,
        "metrics": metrics,
        "history": history,
        "predictions": y_prob.tolist(),
        "client_distribution": [{"client": index, "size": int(len(partition))} for index, partition in enumerate(client_partitions(y_train, config.clients, config.non_iid, config.seed))],
        "dp": {"clip_norm": config.clip_norm, "noise_multiplier": config.noise_multiplier} if config.mode == "dp_fedavg" else None,
    }
