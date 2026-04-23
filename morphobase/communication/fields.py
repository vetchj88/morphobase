import numpy as np


def _smooth_1d(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size <= 1:
        return values.astype(float, copy=True)

    smoothed = np.zeros_like(values, dtype=float)
    for idx in range(values.size):
        left = max(0, idx - radius)
        right = min(values.size, idx + radius + 1)
        smoothed[idx] = float(np.mean(values[left:right]))
    return smoothed


def update_local_field_alignment(hidden: np.ndarray) -> np.ndarray:
    center = hidden.mean(axis=0, keepdims=True)
    diff = np.linalg.norm(hidden - center, axis=1)
    scale = diff.max() if diff.max() > 0 else 1.0
    return 1.0 - diff / scale


def update_tissue_field(
    hidden: np.ndarray,
    membrane: np.ndarray,
    current_tissue_field: np.ndarray,
    z_memory: np.ndarray,
) -> np.ndarray:
    hidden_mean = np.mean(hidden, axis=1)
    local_hidden = _smooth_1d(hidden_mean, radius=1)
    regional_hidden = _smooth_1d(hidden_mean, radius=max(2, hidden.shape[0] // 8))
    local_membrane = _smooth_1d(membrane, radius=1)
    positions = np.linspace(-1.0, 1.0, hidden.shape[0], dtype=float)
    tissue_target = np.tanh(
        0.34 * regional_hidden
        + 0.26 * local_hidden
        + 0.18 * z_memory
        + 0.14 * local_membrane
        + 0.12 * positions
    )
    updated = 0.42 * current_tissue_field + 0.58 * tissue_target
    return np.clip(updated, -1.0, 1.0)
