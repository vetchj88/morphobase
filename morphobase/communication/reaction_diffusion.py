import numpy as np


def _laplacian_1d(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)

    left = np.roll(values, 1)
    right = np.roll(values, -1)
    left[0] = values[0]
    right[-1] = values[-1]
    return left - 2.0 * values + right


def update_reaction_diffusion(
    hidden: np.ndarray,
    tissue_field: np.ndarray,
    oscillation_amplitude: np.ndarray,
    current_activator: np.ndarray,
    current_inhibitor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_mean = np.mean(hidden, axis=1)
    activator_drive = np.clip(0.5 * (np.tanh(hidden_mean) + 1.0), 0.0, 1.0)
    tissue_drive = np.clip(0.5 * (tissue_field + 1.0), 0.0, 1.0)
    oscillation_drive = np.clip(oscillation_amplitude, 0.0, 1.0)

    activator_laplacian = _laplacian_1d(current_activator)
    inhibitor_laplacian = _laplacian_1d(current_inhibitor)

    activator_reaction = (
        0.14 * activator_drive
        + 0.10 * tissue_drive
        + 0.06 * oscillation_drive
        - 0.18 * current_activator
        - 0.12 * current_inhibitor * current_activator
    )
    inhibitor_reaction = (
        0.08 * current_activator
        + 0.05 * tissue_drive
        - 0.15 * current_inhibitor
    )

    next_activator = current_activator + 0.18 * activator_laplacian + activator_reaction
    next_inhibitor = current_inhibitor + 0.28 * inhibitor_laplacian + inhibitor_reaction
    return np.clip(next_activator, 0.0, 1.0), np.clip(next_inhibitor, 0.0, 1.0)


def reaction_diffusion_pattern_strength(activator: np.ndarray, inhibitor: np.ndarray) -> float:
    if activator.size == 0:
        return 0.0
    pattern = activator - inhibitor
    return float(np.mean(np.abs(pattern - np.mean(pattern))))
