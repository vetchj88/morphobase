import numpy as np

def low_utility_mask(energy: np.ndarray, commitment: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    return (energy < threshold) & (commitment < 0.1)
