import numpy as np


def consume_baseline_energy(
    energy: np.ndarray,
    plasticity: np.ndarray,
    stress: np.ndarray | None = None,
    field_alignment: np.ndarray | None = None,
    z_alignment: np.ndarray | None = None,
    growth_pressure: np.ndarray | None = None,
    base_cost: float = 0.0015,
) -> np.ndarray:
    stress = np.zeros_like(energy) if stress is None else stress
    field_alignment = np.zeros_like(energy) if field_alignment is None else field_alignment
    z_alignment = np.zeros_like(energy) if z_alignment is None else z_alignment
    growth_pressure = np.zeros_like(energy) if growth_pressure is None else growth_pressure

    z_support = np.clip(0.5 * (z_alignment + 1.0), 0.0, 1.0)
    maintenance = (
        base_cost
        + 0.0015 * plasticity
        + 0.0010 * stress
        + 0.0007 * (1.0 - field_alignment)
        + 0.0008 * growth_pressure
    )
    recovery = 0.0010 * field_alignment + 0.0008 * z_support + 0.0006 * (1.0 - np.clip(stress, 0.0, 1.0))
    return np.clip(energy - maintenance + recovery, 0.0, 1.0)
