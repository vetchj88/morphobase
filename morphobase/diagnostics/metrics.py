from collections import Counter

import numpy as np

from morphobase.communication.conductance import conductance_entropy
from morphobase.communication.oscillations import oscillation_phase_coherence
from morphobase.communication.predictive_coding import predictive_error_contrast
from morphobase.communication.reaction_diffusion import reaction_diffusion_pattern_strength
from morphobase.communication.stigmergy import stigmergic_highway_strength
from morphobase.communication.z_field import z_field_drift
from morphobase.development.renewal import plasticity_health
from morphobase.organism.state import OrganismState


def stage_occupancy(stages):
    c = Counter(stages.tolist())
    total = max(len(stages), 1)
    return {k: v / total for k, v in c.items()}


def _safe_abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(corr):
        return 0.0
    return abs(corr)


def summarize_state(state: OrganismState, z_history=None) -> dict:
    z_history = z_history or [state.z_alignment]
    return {
        'step_count': state.step_count,
        'mean_energy': float(state.energy.mean()),
        'energy_variance': float(state.energy.var()),
        'mean_stress': float(state.stress.mean()),
        'stress_variance': float(state.stress.var()),
        'mean_plasticity': float(state.plasticity.mean()),
        'mean_commitment': float(state.commitment.mean()),
        'mean_field_alignment': float(state.field_alignment.mean()),
        'mean_tissue_field': float(state.tissue_field.mean()),
        'tissue_field_variance': float(state.tissue_field.var()),
        'tissue_field_regionality': float(np.mean(np.abs(state.tissue_field - np.mean(state.tissue_field)))),
        'tissue_field_z_coupling': _safe_abs_correlation(state.tissue_field, state.z_memory),
        'mean_oscillation_amplitude': float(state.oscillation_amplitude.mean()),
        'oscillation_amplitude_variance': float(state.oscillation_amplitude.var()),
        'oscillation_phase_coherence': oscillation_phase_coherence(state.oscillation_phase),
        'oscillation_regionality': float(
            np.mean(np.abs(state.oscillation_amplitude - np.mean(state.oscillation_amplitude)))
        ),
        'oscillation_tissue_coupling': _safe_abs_correlation(state.oscillation_amplitude, state.tissue_field),
        'mean_morphogen_activator': float(state.morphogen_activator.mean()),
        'mean_morphogen_inhibitor': float(state.morphogen_inhibitor.mean()),
        'reaction_diffusion_pattern_strength': reaction_diffusion_pattern_strength(
            state.morphogen_activator,
            state.morphogen_inhibitor,
        ),
        'reaction_diffusion_balance': float(np.mean(state.morphogen_activator - state.morphogen_inhibitor)),
        'reaction_diffusion_tissue_coupling': _safe_abs_correlation(
            state.morphogen_activator - state.morphogen_inhibitor,
            state.tissue_field,
        ),
        'mean_highway_trace': float(state.highway_trace.mean()),
        'mean_highway_flux': float(state.highway_flux.mean()),
        'stigmergic_highway_strength': stigmergic_highway_strength(
            state.highway_trace,
            state.highway_flux,
        ),
        'stigmergic_tissue_coupling': _safe_abs_correlation(state.highway_trace, state.tissue_field),
        'stigmergic_flux_alignment': _safe_abs_correlation(state.highway_flux, state.field_alignment),
        'mean_predictive_prediction': float(state.predictive_prediction.mean()),
        'mean_predictive_error': float(state.predictive_error.mean()),
        'mean_predictive_precision': float(state.predictive_precision.mean()),
        'predictive_error_contrast': predictive_error_contrast(
            state.predictive_error,
            state.predictive_precision,
        ),
        'predictive_tissue_coupling': _safe_abs_correlation(state.predictive_prediction, state.tissue_field),
        'mean_z_alignment': float(state.z_alignment.mean()),
        'mean_z_memory': float(state.z_memory.mean()),
        'z_memory_alignment_gap': float(np.mean(np.abs(state.z_alignment - state.z_memory))),
        'mean_growth_pressure': float(state.growth_pressure.mean()),
        'mean_growth_cooldown': float(state.growth_cooldown.mean()),
        'mean_growth_activity': float(state.growth_activity.mean()),
        'recent_growth_signal_mean': float(state.recent_growth_signal_mean),
        'recent_growth_event_fraction': float(state.recent_growth_event_fraction),
        'recent_growth_energy_transferred': float(state.recent_growth_energy_transferred),
        'recent_growth_repair_fraction': float(state.recent_growth_repair_fraction),
        'recent_growth_bottleneck_fraction': float(state.recent_growth_bottleneck_fraction),
        'recent_growth_decorative_fraction': float(state.recent_growth_decorative_fraction),
        'recent_structural_churn': float(state.recent_structural_churn),
        'conductance_entropy': conductance_entropy(state.conductance),
        'active_cell_count': int(state.alive.sum()),
        'z_field_drift': z_field_drift(z_history),
        **plasticity_health(state.plasticity, state.commitment, state.energy),
        **{f'stage_{k}': v for k, v in stage_occupancy(state.stages).items()},
    }


def lightcone_proxy(history):
    if len(history) < 2:
        return 0.0
    deltas = [np.mean(np.abs(history[i].hidden - history[i - 1].hidden)) for i in range(1, len(history))]
    return float(np.mean(deltas))
