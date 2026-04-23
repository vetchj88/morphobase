import numpy as np

def diffuse_stress(stress: np.ndarray, conductance: np.ndarray, coefficient: float = 0.1) -> np.ndarray:
    weights = conductance / np.maximum(conductance.sum(axis=1, keepdims=True), 1e-8)
    return np.clip((1 - coefficient) * stress + coefficient * (weights @ stress), 0.0, 5.0)
