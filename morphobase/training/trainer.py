from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphobase.training.losses import softmax_cross_entropy


@dataclass(slots=True)
class PrototypeModel:
    labels: np.ndarray
    centroids: np.ndarray

    def _restricted_centroids(self, allowed_classes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if allowed_classes is None:
            return self.labels, self.centroids
        allowed_classes = np.asarray(allowed_classes, dtype=int)
        indices = np.array([int(np.where(self.labels == label)[0][0]) for label in allowed_classes], dtype=int)
        return allowed_classes, self.centroids[indices]

    def predict(self, embeddings: np.ndarray, allowed_classes: np.ndarray | None = None) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=float)
        labels, centroids = self._restricted_centroids(allowed_classes=allowed_classes)
        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        return labels[np.argmin(distances, axis=1)]

    def score(self, embeddings: np.ndarray, labels: np.ndarray, allowed_classes: np.ndarray | None = None) -> float:
        predictions = self.predict(embeddings, allowed_classes=allowed_classes)
        labels = np.asarray(labels)
        return float(np.mean(predictions == labels))

    def mean_margin(self, embeddings: np.ndarray, allowed_classes: np.ndarray | None = None) -> float:
        embeddings = np.asarray(embeddings, dtype=float)
        _, centroids = self._restricted_centroids(allowed_classes=allowed_classes)
        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        if distances.shape[1] < 2:
            return 0.0
        sorted_distances = np.sort(distances, axis=1)
        margins = sorted_distances[:, 1] - sorted_distances[:, 0]
        return float(np.mean(margins))


class Trainer:
    def train_step(self, embeddings: np.ndarray, labels: np.ndarray) -> PrototypeModel:
        embeddings = np.asarray(embeddings, dtype=float)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        centroids = np.stack([embeddings[labels == label].mean(axis=0) for label in unique_labels], axis=0)
        return PrototypeModel(labels=unique_labels, centroids=centroids)


@dataclass(slots=True)
class LinearClassifierModel:
    class_labels: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    def _restricted_logits(self, embeddings: np.ndarray, allowed_classes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        embeddings = np.asarray(embeddings, dtype=float)
        class_labels = self.class_labels if allowed_classes is None else np.asarray(allowed_classes, dtype=int)
        indices = np.array([int(np.where(self.class_labels == label)[0][0]) for label in class_labels], dtype=int)
        logits = embeddings @ self.weights[:, indices] + self.bias[indices]
        return logits, class_labels

    def predict(self, embeddings: np.ndarray, allowed_classes: np.ndarray | None = None) -> np.ndarray:
        logits, class_labels = self._restricted_logits(embeddings, allowed_classes=allowed_classes)
        return class_labels[np.argmax(logits, axis=1)]

    def score(self, embeddings: np.ndarray, labels: np.ndarray, allowed_classes: np.ndarray | None = None) -> float:
        predictions = self.predict(embeddings, allowed_classes=allowed_classes)
        labels = np.asarray(labels, dtype=int)
        return float(np.mean(predictions == labels))

    def mean_margin(self, embeddings: np.ndarray, allowed_classes: np.ndarray | None = None) -> float:
        logits, _ = self._restricted_logits(embeddings, allowed_classes=allowed_classes)
        if logits.shape[1] < 2:
            return 0.0
        sorted_logits = np.sort(logits, axis=1)
        return float(np.mean(sorted_logits[:, -1] - sorted_logits[:, -2]))


class SequentialLinearTrainer:
    def __init__(self, class_labels: np.ndarray, feature_dim: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.class_labels = np.asarray(class_labels, dtype=int)
        self.label_to_index = {int(label): idx for idx, label in enumerate(self.class_labels)}
        self.weights = rng.normal(0.0, 0.02, size=(feature_dim, self.class_labels.size))
        self.bias = np.zeros(self.class_labels.size, dtype=float)

    def train_task(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 80,
        learning_rate: float = 0.20,
        l2: float = 1e-4,
    ) -> LinearClassifierModel:
        embeddings = np.asarray(embeddings, dtype=float)
        label_indices = np.array([self.label_to_index[int(label)] for label in labels], dtype=int)

        for _ in range(epochs):
            logits = embeddings @ self.weights + self.bias
            _, grad = softmax_cross_entropy(logits, label_indices)
            weight_grad = embeddings.T @ grad + l2 * self.weights
            bias_grad = np.sum(grad, axis=0)
            self.weights -= learning_rate * weight_grad
            self.bias -= learning_rate * bias_grad

        return LinearClassifierModel(
            class_labels=self.class_labels.copy(),
            weights=self.weights.copy(),
            bias=self.bias.copy(),
        )
