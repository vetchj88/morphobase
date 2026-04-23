import numpy as np

from morphobase.cells.local_model import local_prediction


def update_predictive_coding(
    hidden: np.ndarray,
    conductance: np.ndarray,
    tissue_field: np.ndarray,
    z_memory: np.ndarray,
    current_prediction: np.ndarray,
    current_error: np.ndarray,
    current_precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_hidden = local_prediction(hidden)
    weights = np.clip(conductance, 0.0, None)
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    neighbor_hidden = weights @ predicted_hidden / row_sums

    local_signal = np.mean(predicted_hidden, axis=1)
    neighbor_signal = np.mean(neighbor_hidden, axis=1)
    target_prediction = np.tanh(
        0.46 * local_signal
        + 0.28 * neighbor_signal
        + 0.14 * tissue_field
        + 0.12 * z_memory
    )
    next_prediction = np.clip(0.68 * current_prediction + 0.32 * target_prediction, -1.0, 1.0)
    next_error = np.clip(
        0.52 * current_error + 0.48 * np.abs(local_signal - next_prediction),
        0.0,
        1.0,
    )
    precision_target = np.clip(1.0 - next_error + 0.20 * np.clip(np.abs(tissue_field), 0.0, 1.0), 0.0, 1.0)
    next_precision = np.clip(0.62 * current_precision + 0.38 * precision_target, 0.0, 1.0)
    return next_prediction, next_error, next_precision


def predictive_error_contrast(error: np.ndarray, precision: np.ndarray) -> float:
    if error.size == 0:
        return 0.0
    contrast = error * (1.0 - precision)
    return float(np.mean(np.abs(contrast - np.mean(contrast))))
