import numpy as np

def maturity_score(commitment: np.ndarray, stress: np.ndarray, z_alignment: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * commitment + 0.3 * (1 - stress / 5.0) + 0.2 * np.abs(z_alignment), 0.0, 1.0)
