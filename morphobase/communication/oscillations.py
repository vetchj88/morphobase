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


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def update_oscillatory_coupling(
    hidden: np.ndarray,
    membrane: np.ndarray,
    conductance: np.ndarray,
    tissue_field: np.ndarray,
    stress: np.ndarray,
    current_phase: np.ndarray,
    current_amplitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_mean = np.mean(hidden, axis=1)
    local_hidden = _smooth_1d(hidden_mean, radius=1)
    local_membrane = _smooth_1d(membrane, radius=1)
    local_tissue = _smooth_1d(tissue_field, radius=max(2, hidden.shape[0] // 10))
    weights = np.clip(conductance, 0.0, None)
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    phase_vector = np.exp(1j * current_phase)
    neighbor_complex = (weights @ phase_vector[:, None]).ravel() / row_sums.ravel()
    neighbor_phase = np.angle(neighbor_complex)
    coupling_strength = np.clip(np.mean(weights, axis=1), 0.0, 1.0)

    intrinsic_frequency = (
        0.12
        + 0.05 * np.tanh(local_hidden)
        + 0.04 * local_tissue
        + 0.03 * np.tanh(local_membrane)
        - 0.03 * np.clip(stress, 0.0, 1.0)
    )
    phase_step = intrinsic_frequency + 0.18 * coupling_strength * np.sin(neighbor_phase - current_phase)
    updated_phase = _wrap_phase(current_phase + phase_step)

    amplitude_target = np.clip(
        0.14
        + 0.24 * np.abs(local_membrane)
        + 0.22 * np.abs(local_tissue)
        + 0.14 * np.abs(np.sin(updated_phase))
        + 0.10 * np.clip(coupling_strength, 0.0, 1.0)
        - 0.14 * np.clip(stress, 0.0, 1.0),
        0.0,
        1.0,
    )
    updated_amplitude = np.clip(0.58 * current_amplitude + 0.42 * amplitude_target, 0.0, 1.0)
    return updated_phase, updated_amplitude


def oscillation_phase_coherence(phase: np.ndarray) -> float:
    if phase.size == 0:
        return 0.0
    return float(abs(np.mean(np.exp(1j * phase))))
