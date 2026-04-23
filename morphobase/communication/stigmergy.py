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


def update_stigmergic_highways(
    hidden: np.ndarray,
    conductance: np.ndarray,
    field_alignment: np.ndarray,
    energy: np.ndarray,
    current_trace: np.ndarray,
    current_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_activity = np.clip(np.mean(np.abs(hidden), axis=1), 0.0, 2.0) / 2.0
    hidden_activity = np.clip(hidden_activity - 0.08, 0.0, 1.0)
    traffic = np.clip(np.mean(np.clip(conductance, 0.0, None), axis=1) - 0.18, 0.0, 1.0)
    field_drive = np.clip(field_alignment - 0.30, 0.0, 1.0)
    energy_support = np.clip(energy - 0.35, 0.0, 1.0)

    deposition = np.clip(
        0.22 * hidden_activity
        + 0.30 * traffic
        + 0.26 * field_drive
        + 0.12 * energy_support,
        0.0,
        1.0,
    )
    smoothed_trace = _smooth_1d(current_trace, radius=max(2, hidden.shape[0] // 10))
    next_trace = np.clip(
        0.68 * current_trace
        + 0.12 * smoothed_trace
        + 0.16 * deposition,
        0.0,
        1.0,
    )
    next_flux = np.clip(
        0.50 * current_flux
        + 0.36 * np.sqrt(np.clip(next_trace * (0.15 + traffic) * (0.15 + field_drive), 0.0, 1.0)),
        0.0,
        1.0,
    )
    return next_trace, next_flux


def stigmergic_highway_strength(trace: np.ndarray, flux: np.ndarray) -> float:
    if trace.size == 0:
        return 0.0
    combined = 0.65 * trace + 0.35 * flux
    return float(np.mean(np.abs(combined - np.mean(combined))))
