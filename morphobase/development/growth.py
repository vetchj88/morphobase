import numpy as np

from morphobase.metabolism.budgets import growth_budget, reserve_margin
from morphobase.types import Stage


def growth_pressure(
    stress: np.ndarray,
    field_alignment: np.ndarray,
    z_alignment: np.ndarray,
    predictive_error: np.ndarray | None = None,
    cooldown: np.ndarray | None = None,
) -> np.ndarray:
    predictive_error = np.zeros_like(stress) if predictive_error is None else np.clip(predictive_error, 0.0, 1.0)
    cooldown = np.zeros_like(stress) if cooldown is None else np.clip(cooldown, 0.0, 1.0)
    z_support = np.clip(0.5 * (z_alignment + 1.0), 0.0, 1.0)
    base_pressure = 0.45 * stress + 0.22 * (1.0 - field_alignment) + 0.13 * (1.0 - z_support)
    mismatch_pressure = 0.20 * predictive_error
    hysteresis = 0.28 * cooldown
    return np.clip(base_pressure + mismatch_pressure - hysteresis, 0.0, 1.0)


def should_grow(
    mean_stress: float,
    mean_energy: float,
    mean_growth_pressure: float,
    active_need_fraction: float,
    threshold: float = 0.30,
) -> bool:
    return mean_growth_pressure > threshold and mean_energy > 0.20 and mean_stress > 0.03 and active_need_fraction > 0.15


def apply_regulated_growth(
    energy: np.ndarray,
    stress: np.ndarray,
    plasticity: np.ndarray,
    commitment: np.ndarray,
    field_alignment: np.ndarray,
    z_alignment: np.ndarray,
    predictive_error: np.ndarray,
    cooldown: np.ndarray,
    activity: np.ndarray,
    reserve_floor: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    pressure = growth_pressure(stress, field_alignment, z_alignment, predictive_error=predictive_error, cooldown=cooldown)
    z_support = np.clip(0.5 * (z_alignment + 1.0), 0.0, 1.0)
    energy_deficit = np.clip((0.45 - energy) / 0.45, 0.0, 1.0)
    repair_need = np.clip(
        0.48 * stress + 0.20 * (1.0 - field_alignment) + 0.14 * (1.0 - z_support) + 0.18 * energy_deficit,
        0.0,
        1.0,
    )
    bottleneck_need = np.clip(
        0.46 * predictive_error + 0.24 * (1.0 - field_alignment) + 0.12 * (1.0 - z_support) + 0.18 * energy_deficit,
        0.0,
        1.0,
    )
    need_score = np.maximum(repair_need, bottleneck_need)
    active_need_fraction = float(np.mean(need_score > 0.35))
    next_cooldown = np.clip(cooldown - 0.12, 0.0, 1.0)
    next_activity = np.clip(activity * 0.72, 0.0, 1.0)
    mean_pressure = float(pressure.mean())
    if not should_grow(float(stress.mean()), float(energy.mean()), mean_pressure, active_need_fraction):
        return energy, plasticity, commitment, pressure, next_cooldown, next_activity, {
            'growth_signal_mean': mean_pressure,
            'growth_need_fraction': active_need_fraction,
            'growth_event_fraction': 0.0,
            'growth_energy_transferred': 0.0,
            'growth_repair_fraction': 0.0,
            'growth_bottleneck_fraction': 0.0,
            'growth_decorative_fraction': 0.0,
            'structural_churn': float(np.mean(np.abs(next_activity - activity))),
        }

    repair_threshold = float(np.quantile(repair_need, 0.70))
    bottleneck_threshold = float(np.quantile(bottleneck_need, 0.70))
    recipients = (
        (pressure >= max(0.34, float(np.quantile(pressure, 0.60))))
        & (cooldown < 0.35)
        & ((repair_need >= repair_threshold) | (bottleneck_need >= bottleneck_threshold))
    )
    donor_surplus = reserve_margin(energy, reserve_floor=reserve_floor)
    donors = donor_surplus > 0.0
    recipient_budget = growth_budget(energy, stress, field_alignment, reserve_floor=reserve_floor, mobilization=0.14)
    need_weights = 0.65 * repair_need + 0.35 * bottleneck_need
    recipient_budget = recipient_budget * (0.70 * need_weights + 0.30 * energy_deficit)

    if not recipients.any() or not donors.any() or float(recipient_budget.sum()) <= 0.0:
        return energy, plasticity, commitment, pressure, next_cooldown, next_activity, {
            'growth_signal_mean': mean_pressure,
            'growth_need_fraction': active_need_fraction,
            'growth_event_fraction': float(recipients.mean()),
            'growth_energy_transferred': 0.0,
            'growth_repair_fraction': 0.0,
            'growth_bottleneck_fraction': 0.0,
            'growth_decorative_fraction': 0.0,
            'structural_churn': float(np.mean(np.abs(next_activity - activity))),
        }

    transfer_pool = min(float(donor_surplus.sum()) * 0.10, float(recipient_budget.sum()) * 0.75)
    if transfer_pool <= 0.0:
        return energy, plasticity, commitment, pressure, next_cooldown, next_activity, {
            'growth_signal_mean': mean_pressure,
            'growth_need_fraction': active_need_fraction,
            'growth_event_fraction': float(recipients.mean()),
            'growth_energy_transferred': 0.0,
            'growth_repair_fraction': 0.0,
            'growth_bottleneck_fraction': 0.0,
            'growth_decorative_fraction': 0.0,
            'structural_churn': float(np.mean(np.abs(next_activity - activity))),
        }

    donor_weights = donor_surplus[donors]
    donor_weights = donor_weights / max(float(donor_weights.sum()), 1e-8)

    recipient_weights = recipient_budget[recipients]
    recipient_weights = recipient_weights / max(float(recipient_weights.sum()), 1e-8)

    next_energy = energy.copy()
    next_plasticity = plasticity.copy()
    next_commitment = commitment.copy()

    next_energy[donors] -= transfer_pool * donor_weights
    next_energy[recipients] += transfer_pool * recipient_weights

    rescue_signal = transfer_pool * recipient_weights
    next_plasticity[recipients] = np.clip(next_plasticity[recipients] + 0.18 * rescue_signal, 0.0, 1.0)
    next_commitment[recipients] = np.clip(next_commitment[recipients] - 0.08 * rescue_signal, 0.0, 1.0)
    next_energy = np.clip(next_energy, 0.0, 1.0)

    recipient_indices = np.flatnonzero(recipients)
    if recipient_indices.size:
        rescue_scale = rescue_signal / max(float(np.max(rescue_signal)), 1e-8)
        next_activity[recipient_indices] = np.clip(
            next_activity[recipient_indices] + 0.45 * rescue_scale,
            0.0,
            1.0,
        )
        next_cooldown[recipient_indices] = np.clip(
            next_cooldown[recipient_indices] + 0.55 * rescue_scale,
            0.0,
            1.0,
        )

    repair_fraction = float(np.mean(repair_need[recipients] >= bottleneck_need[recipients])) if recipient_indices.size else 0.0
    bottleneck_fraction = float(np.mean(bottleneck_need[recipients] > repair_need[recipients])) if recipient_indices.size else 0.0
    decorative_fraction = float(np.mean((need_score[recipients] < 0.32))) if recipient_indices.size else 0.0
    structural_churn = float(np.mean(np.abs(next_activity - activity)))

    return next_energy, next_plasticity, next_commitment, pressure, next_cooldown, next_activity, {
        'growth_signal_mean': mean_pressure,
        'growth_need_fraction': active_need_fraction,
        'growth_event_fraction': float(recipients.mean()),
        'growth_energy_transferred': transfer_pool,
        'growth_repair_fraction': repair_fraction,
        'growth_bottleneck_fraction': bottleneck_fraction,
        'growth_decorative_fraction': decorative_fraction,
        'structural_churn': structural_churn,
    }


def assign_stages(
    plasticity: np.ndarray,
    commitment: np.ndarray,
    energy: np.ndarray,
    stress: np.ndarray,
) -> np.ndarray:
    stages = np.full(plasticity.shape[0], Stage.SEED.value, dtype=object)
    stages[(plasticity > 0.65) & (commitment < 0.2)] = Stage.EXPLORATORY.value
    stages[(plasticity > 0.3) & (commitment >= 0.2)] = Stage.DIFFERENTIATING.value
    stages[(commitment > 0.7) & (plasticity >= 0.15) & (energy > 0.25)] = Stage.MATURE.value
    stages[(stress > 0.4) | (energy < 0.18)] = Stage.DEDIFFERENTIATING.value
    stages[(plasticity < 0.08) | (energy < 0.08)] = Stage.PRUNABLE.value
    return stages
