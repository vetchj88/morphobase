import numpy as np


def update_z_memory(
    hidden: np.ndarray,
    current_alignment: np.ndarray,
    current_memory: np.ndarray,
    memory_rate: float = 0.0125,
) -> np.ndarray:
    target = np.tanh(hidden.mean(axis=1))
    desired_memory = 0.7 * current_memory + 0.3 * (0.5 * current_alignment + 0.5 * target)
    return np.clip((1 - memory_rate) * current_memory + memory_rate * desired_memory, -1.0, 1.0)


def update_z_alignment(
    hidden: np.ndarray,
    current: np.ndarray,
    memory: np.ndarray,
    rate: float = 0.05,
) -> np.ndarray:
    target = np.tanh(hidden.mean(axis=1))
    blended_target = 0.65 * memory + 0.35 * target
    return np.clip((1 - rate) * current + rate * blended_target, -1.0, 1.0)

def z_field_drift(history: list[np.ndarray]) -> float:
    if len(history) < 2:
        return 0.0
    return float(np.mean(np.abs(history[-1] - history[0])))
