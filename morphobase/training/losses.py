import numpy as np

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred-target)**2))


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    loss = -np.mean(np.log(np.clip(probs[np.arange(labels.size), labels], 1e-8, 1.0)))
    grad = probs
    grad[np.arange(labels.size), labels] -= 1.0
    grad /= max(labels.size, 1)
    return float(loss), grad
