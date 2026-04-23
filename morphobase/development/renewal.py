import numpy as np


def plasticity_health(
    plasticity: np.ndarray,
    commitment: np.ndarray,
    energy: np.ndarray | None = None,
) -> dict[str, float]:
    energy = np.ones_like(plasticity) if energy is None else energy
    dormant = plasticity < 0.05
    active = plasticity > 0.2
    mature = commitment > 0.7
    pseudo_mature = (plasticity < 0.12) & (commitment > 0.55)
    plasticity_loss = (plasticity < 0.12) & (commitment > 0.35)
    return {
        'dormant_fraction': float(np.mean(dormant)),
        'active_fraction': float(np.mean(active)),
        'mature_fraction': float(np.mean(mature)),
        'plasticity_loss_index': float(np.mean(plasticity_loss)),
        'pseudo_maturity_index': float(np.mean(pseudo_mature)),
        'low_energy_fraction': float(np.mean(energy < 0.15)),
    }
