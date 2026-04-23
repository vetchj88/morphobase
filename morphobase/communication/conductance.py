import numpy as np

def conductance_entropy(conductance: np.ndarray, eps: float = 1e-8) -> float:
    flat = conductance.flatten()
    flat = flat / max(flat.sum(), eps)
    return float(-(flat * np.log(flat + eps)).sum())

def update_conductance(conductance: np.ndarray, membrane: np.ndarray, stress: np.ndarray) -> np.ndarray:
    dv = np.abs(membrane[:, None] - membrane[None, :])
    ds = np.abs(stress[:, None] - stress[None, :])
    out = np.exp(-(dv + 0.5 * ds))
    np.fill_diagonal(out, 1.0)
    return out
