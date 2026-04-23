import numpy as np


def transition_affordable(energy: float, cost: float) -> bool:
    return energy >= cost


def reserve_margin(energy: np.ndarray, reserve_floor: float = 0.15) -> np.ndarray:
    return np.clip(energy - reserve_floor, 0.0, None)


def growth_budget(
    energy: np.ndarray,
    stress: np.ndarray,
    field_alignment: np.ndarray,
    reserve_floor: float = 0.15,
    mobilization: float = 0.2,
) -> np.ndarray:
    available = reserve_margin(energy, reserve_floor=reserve_floor)
    if float(available.sum()) <= 0.0:
        return np.zeros_like(energy)

    demand = np.clip(0.65 * stress + 0.35 * (1.0 - field_alignment), 0.0, 1.0)
    total_demand = float(demand.sum())
    if total_demand <= 0.0:
        return np.zeros_like(energy)

    pool = float(available.sum()) * mobilization
    return pool * (demand / total_demand)
