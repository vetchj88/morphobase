from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        max_seq_len: int,
        num_classes: int,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 96,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.input_proj = nn.Linear(int(input_dim), int(d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, self.max_seq_len + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, int(num_classes))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [batch, seq, dim], got {tuple(tokens.shape)}")
        batch_size, seq_len, _ = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")
        projected = self.input_proj(tokens)
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = torch.cat([cls, projected], dim=1)
        encoded = encoded + self.position[:, : seq_len + 1, :]
        encoded = self.encoder(encoded)
        pooled = self.norm(encoded[:, 0, :])
        return self.head(pooled)


@dataclass(slots=True)
class TransformerRunSummary:
    parameter_count: int
    train_wall_time_sec: float
    eval_wall_time_sec: float
    train_samples_per_sec: float
    eval_samples_per_sec: float


@dataclass(slots=True)
class StaticPrototypeModel:
    labels: np.ndarray
    centroids: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float32)
        distances = np.linalg.norm(feats[:, None, :] - self.centroids[None, :, :], axis=2)
        return self.labels[np.argmin(distances, axis=1)]


class TinyMLPClassifier(nn.Module):
    def __init__(self, *, input_dim: int, num_classes: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected features with shape [batch, dim], got {tuple(features.shape)}")
        return self.network(features)


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def fit_static_prototype(features: np.ndarray, labels: np.ndarray) -> tuple[StaticPrototypeModel, TransformerRunSummary]:
    feats = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(label_array)
    start = perf_counter()
    centroids = np.stack([feats[label_array == label].mean(axis=0) for label in unique_labels], axis=0)
    elapsed = perf_counter() - start
    return StaticPrototypeModel(labels=unique_labels, centroids=centroids), TransformerRunSummary(
        parameter_count=int(centroids.size),
        train_wall_time_sec=float(elapsed),
        eval_wall_time_sec=0.0,
        train_samples_per_sec=float(len(feats) / max(elapsed, 1e-8)),
        eval_samples_per_sec=0.0,
    )


def score_static_prototype(
    model: StaticPrototypeModel,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, TransformerRunSummary]:
    feats = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    start = perf_counter()
    predictions = model.predict(feats)
    elapsed = perf_counter() - start
    accuracy = float(np.mean(predictions == label_array))
    distances = np.linalg.norm(feats[:, None, :] - model.centroids[None, :, :], axis=2)
    if distances.shape[1] < 2:
        margin = 0.0
    else:
        sorted_distances = np.sort(distances, axis=1)
        margin = float(np.mean(sorted_distances[:, 1] - sorted_distances[:, 0]))
    return accuracy, margin, TransformerRunSummary(
        parameter_count=int(model.centroids.size),
        train_wall_time_sec=0.0,
        eval_wall_time_sec=float(elapsed),
        train_samples_per_sec=0.0,
        eval_samples_per_sec=float(len(feats) / max(elapsed, 1e-8)),
    )


def fit_mlp_classifier(
    model: TinyMLPClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 60,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cpu",
) -> TransformerRunSummary:
    feats = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    torch.manual_seed(int(seed))
    dataset = TensorDataset(torch.from_numpy(feats), torch.from_numpy(label_array))
    loader = DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=True)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    start = perf_counter()
    for _ in range(int(epochs)):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
    elapsed = perf_counter() - start
    return TransformerRunSummary(
        parameter_count=parameter_count(model),
        train_wall_time_sec=float(elapsed),
        eval_wall_time_sec=0.0,
        train_samples_per_sec=float(len(feats) * max(int(epochs), 1) / max(elapsed, 1e-8)),
        eval_samples_per_sec=0.0,
    )


def score_mlp_classifier(
    model: TinyMLPClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[float, float, TransformerRunSummary]:
    feats = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(feats))
    loader = DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=False)
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    start = perf_counter()
    with torch.no_grad():
        for (batch_features,) in loader:
            logits = model(batch_features.to(device))
            outputs.append(logits.cpu().numpy())
    elapsed = perf_counter() - start
    logits_np = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)
    predictions = np.argmax(logits_np, axis=1) if logits_np.size else np.zeros_like(label_array)
    accuracy = float(np.mean(predictions == label_array)) if label_array.size else 0.0
    if logits_np.shape[1] < 2:
        margin = 0.0
    else:
        sorted_logits = np.sort(logits_np, axis=1)
        margin = float(np.mean(sorted_logits[:, -1] - sorted_logits[:, -2]))
    return accuracy, margin, TransformerRunSummary(
        parameter_count=parameter_count(model),
        train_wall_time_sec=0.0,
        eval_wall_time_sec=float(elapsed),
        train_samples_per_sec=0.0,
        eval_samples_per_sec=float(len(feats) / max(elapsed, 1e-8)),
    )


def fit_transformer_classifier(
    model: TinyTransformerClassifier,
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 40,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cpu",
) -> TransformerRunSummary:
    tokens_np = np.asarray(tokens, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    torch.manual_seed(int(seed))
    dataset = TensorDataset(torch.from_numpy(tokens_np), torch.from_numpy(labels_np))
    loader = DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=True)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    start = perf_counter()
    for _ in range(int(epochs)):
        for batch_tokens, batch_labels in loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tokens)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
    elapsed = perf_counter() - start
    train_rate = float(len(dataset) * max(int(epochs), 1) / max(elapsed, 1e-8))
    return TransformerRunSummary(
        parameter_count=parameter_count(model),
        train_wall_time_sec=float(elapsed),
        eval_wall_time_sec=0.0,
        train_samples_per_sec=train_rate,
        eval_samples_per_sec=0.0,
    )


def predict_transformer_classifier(
    model: TinyTransformerClassifier,
    tokens: np.ndarray,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, TransformerRunSummary]:
    tokens_np = np.asarray(tokens, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(tokens_np))
    loader = DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=False)
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    start = perf_counter()
    with torch.no_grad():
        for (batch_tokens,) in loader:
            logits = model(batch_tokens.to(device))
            outputs.append(logits.cpu().numpy())
    elapsed = perf_counter() - start
    logits_np = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)
    eval_rate = float(len(tokens_np) / max(elapsed, 1e-8))
    return logits_np, TransformerRunSummary(
        parameter_count=parameter_count(model),
        train_wall_time_sec=0.0,
        eval_wall_time_sec=float(elapsed),
        train_samples_per_sec=0.0,
        eval_samples_per_sec=eval_rate,
    )


def score_transformer_classifier(
    model: TinyTransformerClassifier,
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[float, float, TransformerRunSummary]:
    logits, summary = predict_transformer_classifier(model, tokens, batch_size=batch_size, device=device)
    if logits.size == 0:
        return 0.0, 0.0, summary
    labels_np = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == labels_np))
    if logits.shape[1] < 2:
        margin = 0.0
    else:
        sorted_logits = np.sort(logits, axis=1)
        margin = float(np.mean(sorted_logits[:, -1] - sorted_logits[:, -2]))
    return accuracy, margin, summary
