import numpy as np

from morphobase.cells.local_model import local_prediction


def _neighbor_average(hidden: np.ndarray, conductance: np.ndarray) -> np.ndarray:
    weights = np.clip(conductance, 0.0, None)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return weights @ hidden / row_sums


def minimal_cell_update(
    hidden,
    membrane,
    stress,
    plasticity,
    conductance,
    z_alignment,
    field_alignment,
    target_value: float,
    noise_scale: float,
):
    pred = local_prediction(hidden)
    neighbor_hidden = _neighbor_average(pred, conductance)
    mean_hidden = pred.mean(axis=1)
    mean_neighbor = neighbor_hidden.mean(axis=1)
    z_drive = 0.25 * np.clip(z_alignment, -1.0, 1.0)
    field_drive = 0.10 * (np.clip(field_alignment, 0.0, 1.0) - 0.5)
    error = target_value - mean_hidden + z_drive + field_drive
    coordination_mismatch = np.mean(np.abs(pred - neighbor_hidden), axis=1)
    stress = np.clip(0.75 * stress + 0.15 * np.abs(error) + 0.10 * coordination_mismatch, 0.0, 5.0)
    membrane = np.clip(0.85 * membrane + 0.10 * error + 0.05 * (mean_neighbor - mean_hidden), -1.0, 1.0)
    plasticity = np.clip(0.94 * plasticity + 0.04 * (1.0 - stress / 5.0) + 0.02 * np.abs(z_alignment), 0.0, 1.0)
    noise = np.random.normal(0.0, noise_scale, size=hidden.shape)
    hidden = np.clip(
        hidden
        + 0.08 * plasticity[:, None] * error[:, None]
        + 0.06 * (neighbor_hidden - pred)
        + noise,
        -2.0,
        2.0,
    )
    return hidden, membrane, stress, plasticity
