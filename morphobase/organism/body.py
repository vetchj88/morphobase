import numpy as np

from morphobase.cells.femo import minimal_cell_update
from morphobase.communication.conductance import update_conductance
from morphobase.communication.fields import update_local_field_alignment, update_tissue_field
from morphobase.communication.oscillations import update_oscillatory_coupling
from morphobase.communication.predictive_coding import update_predictive_coding
from morphobase.communication.reaction_diffusion import update_reaction_diffusion
from morphobase.communication.stigmergy import update_stigmergic_highways
from morphobase.communication.z_field import update_z_alignment, update_z_memory
from morphobase.development.growth import apply_regulated_growth, assign_stages
from morphobase.metabolism.energy import consume_baseline_energy
from morphobase.organism.state import OrganismState


class Body:
    def __init__(self, state: OrganismState):
        self.state = state

    def _neighbor_hidden(self) -> np.ndarray:
        s = self.state
        weights = np.clip(s.conductance, 0.0, None)
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return weights @ s.hidden / row_sums

    def _spatial_hidden_anchor(self, radius: int = 3) -> np.ndarray:
        s = self.state
        anchor = np.zeros_like(s.hidden)
        num_cells = s.hidden.shape[0]
        for idx in range(num_cells):
            left = max(0, idx - radius)
            right = min(num_cells, idx + radius + 1)
            local = s.hidden[left:right]
            if local.shape[0] <= 1:
                anchor[idx] = s.hidden[idx]
                continue
            anchor[idx] = (local.sum(axis=0) - s.hidden[idx]) / max(local.shape[0] - 1, 1)
        return anchor

    def _parameter_incoherence_mask(self) -> np.ndarray:
        s = self.state
        neighbor_hidden = self._neighbor_hidden()
        hidden_consensus_gap = np.mean(np.abs(s.hidden - neighbor_hidden), axis=1)
        membrane_neighbor_gap = np.abs(s.membrane - np.mean(neighbor_hidden, axis=1))
        z_memory_gap = np.abs(s.z_alignment - s.z_memory)
        role_disorder = np.std(s.role_logits, axis=1)
        return (
            (role_disorder > 0.75)
            | (((hidden_consensus_gap > 0.38) | (membrane_neighbor_gap > 0.30)) & (s.stress > 0.18))
            | ((z_memory_gap > 0.48) & (role_disorder > 0.55) & (s.stress > 0.18))
        )

    def _repair_parameter_drift(self) -> None:
        s = self.state
        incoherent = self._parameter_incoherence_mask()
        if not np.any(incoherent):
            return

        neighbor_hidden = self._neighbor_hidden()
        hidden_mean = np.mean(s.hidden, axis=1)
        neighbor_mean = np.mean(neighbor_hidden, axis=1)
        role_disorder = np.std(s.role_logits, axis=1)
        role_mean = np.mean(s.role_logits, axis=0, keepdims=True)
        z_target = 0.65 * s.z_memory + 0.35 * np.tanh(hidden_mean)
        membrane_target = np.clip(0.60 * s.membrane + 0.40 * neighbor_mean, -1.0, 1.0)
        correction_gain = np.clip(
            0.10
            + 0.12 * s.plasticity
            + 0.10 * np.clip(s.energy - 0.10, 0.0, 1.0)
            + 0.10 * np.clip(0.5 * (s.z_memory + 1.0), 0.0, 1.0),
            0.0,
            0.38,
        ) * incoherent.astype(float)
        correction_gain = np.clip(
            correction_gain + 0.18 * np.clip(role_disorder - 0.55, 0.0, 1.0),
            0.0,
            0.52,
        )
        if not np.any(correction_gain > 0.0):
            return

        s.hidden = np.clip(
            s.hidden + correction_gain[:, None] * (0.72 * neighbor_hidden + 0.28 * z_target[:, None] - s.hidden),
            -2.0,
            2.0,
        )
        s.membrane = np.clip(
            s.membrane + correction_gain * (membrane_target - s.membrane),
            -1.0,
            1.0,
        )
        s.z_alignment = np.clip(
            s.z_alignment + 0.72 * correction_gain * (z_target - s.z_alignment),
            -1.0,
            1.0,
        )
        s.z_memory = np.clip(
            s.z_memory + 0.45 * correction_gain * (np.tanh(neighbor_mean) - s.z_memory),
            -1.0,
            1.0,
        )
        s.role_logits[incoherent] = (
            0.45 * s.role_logits[incoherent] + 0.55 * role_mean
        )
        s.field_alignment[incoherent] = np.clip(
            s.field_alignment[incoherent] + 0.18 + 0.24 * correction_gain[incoherent],
            0.0,
            1.0,
        )
        s.plasticity[incoherent] = np.clip(s.plasticity[incoherent] + 0.08, 0.0, 1.0)
        s.commitment[incoherent] = np.clip(s.commitment[incoherent] - 0.05, 0.0, 1.0)
        s.energy[incoherent] = np.clip(s.energy[incoherent] + 0.012, 0.0, 1.0)
        s.stress[incoherent] = np.clip(s.stress[incoherent] * 0.92, 0.0, 5.0)

    def _port_disruption_mask(self) -> np.ndarray:
        s = self.state
        num_cells = s.hidden.shape[0]
        port_width = max(4, num_cells // 8)
        repair_corridor = min(num_cells, port_width + max(2, num_cells // 16))
        boundary = np.zeros(num_cells, dtype=bool)
        boundary[:repair_corridor] = True
        boundary[-repair_corridor:] = True
        neighbor_hidden = self._neighbor_hidden()
        hidden_gap = np.mean(np.abs(s.hidden - neighbor_hidden), axis=1)
        z_gap = np.abs(s.z_alignment - s.z_memory)
        coupling = np.mean(np.clip(s.conductance, 0.0, None), axis=1)
        return boundary & (
            ((coupling < 0.68) & ((s.field_alignment < 0.58) | (s.energy < 0.40)))
            | ((hidden_gap > 0.16) & (s.stress > 0.10))
            | ((z_gap > 0.16) & (s.field_alignment < 0.58))
        )

    def _repair_port_regions(self) -> None:
        s = self.state
        disrupted = self._port_disruption_mask()
        if not np.any(disrupted):
            return

        num_cells = s.hidden.shape[0]
        neighbor_hidden = self._neighbor_hidden()
        spatial_anchor = self._spatial_hidden_anchor(radius=3)
        anchor_mean = np.mean(spatial_anchor, axis=1)
        hidden_mean = np.mean(s.hidden, axis=1)
        role_mean = np.mean(s.role_logits, axis=0, keepdims=True)
        z_support = np.clip(0.5 * (s.z_memory + 1.0), 0.0, 1.0)
        coupling = np.mean(np.clip(s.conductance, 0.0, None), axis=1)
        hidden_gap = np.mean(np.abs(s.hidden - neighbor_hidden), axis=1)
        repeated_hit_load = np.clip(
            0.70 * np.clip(0.60 - coupling, 0.0, 1.0)
            + 0.55 * np.clip(s.stress - 0.25, 0.0, 1.0)
            + 0.45 * np.clip(hidden_gap - 0.18, 0.0, 1.0),
            0.0,
            1.0,
        )
        repair_gain = np.clip(
            0.16
            + 0.14 * s.plasticity
            + 0.10 * np.clip(s.energy - 0.08, 0.0, 1.0)
            + 0.08 * z_support
            - 0.05 * np.clip(s.commitment, 0.0, 1.0),
            0.0,
            0.44,
        ) * disrupted.astype(float)
        repair_gain = np.clip(repair_gain + 0.14 * repeated_hit_load * disrupted.astype(float), 0.0, 0.56)
        if not np.any(repair_gain > 0.0):
            return

        repair_target = 0.50 * spatial_anchor + 0.30 * neighbor_hidden + 0.20 * s.z_memory[:, None]
        membrane_target = np.clip(0.58 * anchor_mean + 0.22 * np.tanh(hidden_mean) + 0.20 * s.z_memory, -1.0, 1.0)
        z_target = np.clip(0.72 * s.z_memory + 0.28 * np.tanh(anchor_mean), -1.0, 1.0)

        s.hidden = np.clip(
            s.hidden + repair_gain[:, None] * (repair_target - s.hidden),
            -2.0,
            2.0,
        )
        s.membrane = np.clip(
            s.membrane + 0.85 * repair_gain * (membrane_target - s.membrane),
            -1.0,
            1.0,
        )
        s.z_alignment = np.clip(
            s.z_alignment + 0.80 * repair_gain * (z_target - s.z_alignment),
            -1.0,
            1.0,
        )
        s.field_alignment[disrupted] = np.clip(
            s.field_alignment[disrupted] + 0.18 + 0.22 * repair_gain[disrupted],
            0.0,
            1.0,
        )
        s.role_logits[disrupted] = 0.65 * s.role_logits[disrupted] + 0.35 * role_mean
        s.plasticity[disrupted] = np.clip(s.plasticity[disrupted] + 0.08, 0.0, 1.0)
        s.commitment[disrupted] = np.clip(s.commitment[disrupted] - 0.05, 0.0, 1.0)
        s.energy[disrupted] = np.clip(
            s.energy[disrupted] + 0.024 + 0.016 * z_support[disrupted] + 0.018 * repeated_hit_load[disrupted],
            0.0,
            1.0,
        )
        s.stress[disrupted] = np.clip(
            s.stress[disrupted] * (0.84 - 0.08 * repeated_hit_load[disrupted]),
            0.0,
            5.0,
        )
        s.z_memory[disrupted] = np.clip(
            s.z_memory[disrupted] + 0.10 * repair_gain[disrupted] * (z_target[disrupted] - s.z_memory[disrupted]),
            -1.0,
            1.0,
        )

        support_band = np.zeros(num_cells, dtype=bool)
        for idx in np.flatnonzero(disrupted):
            left = max(0, idx - 2)
            right = min(num_cells, idx + 3)
            support_band[left:right] = True
        donor_band = support_band & (~disrupted)
        if np.any(donor_band):
            transferable = np.clip(s.energy[donor_band] - 0.18, 0.0, None)
            reserve_pool = float(np.sum(transferable))
            demand = float(np.sum(np.clip(0.32 - s.energy[disrupted], 0.0, None)))
            transfer = min(reserve_pool * 0.35, demand)
            if transfer > 1e-8 and demand > 1e-8:
                donor_indices = np.flatnonzero(donor_band)
                disrupted_indices = np.flatnonzero(disrupted)
                donor_weights = transferable / max(np.sum(transferable), 1e-8)
                demand_weights = np.clip(0.32 - s.energy[disrupted], 0.0, None)
                demand_weights = demand_weights / max(np.sum(demand_weights), 1e-8)
                s.energy[donor_indices] = np.clip(
                    s.energy[donor_indices] - transfer * donor_weights,
                    0.0,
                    1.0,
                )
                s.energy[disrupted_indices] = np.clip(
                    s.energy[disrupted_indices] + transfer * demand_weights,
                    0.0,
                    1.0,
                )
                s.field_alignment[donor_indices] = np.clip(
                    s.field_alignment[donor_indices] + 0.04 * donor_weights / max(np.max(donor_weights), 1e-8),
                    0.0,
                    1.0,
                )
                s.stress[donor_indices] = np.clip(s.stress[donor_indices] * 0.96, 0.0, 5.0)

        s.field_alignment[support_band] = np.clip(
            s.field_alignment[support_band] + 0.06 + 0.06 * np.mean(repair_gain[disrupted]),
            0.0,
            1.0,
        )
        s.z_alignment[support_band] = np.clip(
            s.z_alignment[support_band] + 0.05 * np.mean(z_support[disrupted]),
            -1.0,
            1.0,
        )

        disrupted_indices = np.flatnonzero(disrupted)
        if disrupted_indices.size:
            base_distance = np.abs(np.arange(num_cells)[:, None] - np.arange(num_cells)[None, :])
            locality_kernel = np.exp(-base_distance / 2.0)
            row_strength = np.clip(repair_gain + 0.20 * repeated_hit_load, 0.0, 0.75)
            column_strength = row_strength.copy()
            repair_matrix = np.maximum(row_strength[:, None], column_strength[None, :]) * locality_kernel
            boundary_focus = disrupted.astype(float)[:, None] + disrupted.astype(float)[None, :]
            repair_matrix *= np.clip(boundary_focus, 0.0, 1.0)
            coupling_floor = 0.18 + 0.42 * z_support[:, None] * locality_kernel
            updated = np.maximum(
                s.conductance,
                repair_matrix + coupling_floor * repair_matrix,
            )
            s.conductance = np.clip(updated, 0.0, 2.0)
            diagonal = np.diag_indices_from(s.conductance)
            s.conductance[diagonal] = 1.0

    def fast_step(self, noise_scale: float = 0.0, target_value: float = 0.75):
        s = self.state
        s.hidden, s.membrane, s.stress, s.plasticity = minimal_cell_update(
            s.hidden,
            s.membrane,
            s.stress,
            s.plasticity,
            s.conductance,
            s.z_alignment,
            s.field_alignment,
            target_value,
            noise_scale,
        )
        s.energy = consume_baseline_energy(
            s.energy,
            s.plasticity,
            stress=s.stress,
            field_alignment=s.field_alignment,
            z_alignment=s.z_alignment,
            growth_pressure=s.growth_pressure,
        )

    def _repair_distressed_regions(self, *, no_gradient: bool = False):
        s = self.state
        parameter_incoherence = self._parameter_incoherence_mask()
        distressed = (
            (s.stress > 0.28)
            | (s.energy < 0.20)
            | (np.linalg.norm(s.hidden, axis=1) < 0.12)
            | parameter_incoherence
        )
        if not np.any(distressed):
            return

        neighbor_hidden = self._neighbor_hidden()
        memory_target = np.repeat(s.z_memory[:, None], s.hidden.shape[1], axis=1)
        repair_target = 0.7 * neighbor_hidden + 0.3 * memory_target
        z_support = np.clip(0.5 * (s.z_memory + 1.0), 0.0, 1.0)
        repair_gain = np.clip(
            0.10
            + 0.16 * s.plasticity
            + 0.10 * np.clip(s.energy - 0.05, 0.0, 1.0)
            + 0.08 * z_support,
            0.0,
            0.40,
        )
        if no_gradient:
            repair_gain *= 0.82
        repair_gain = np.clip(repair_gain + 0.06 * parameter_incoherence.astype(float), 0.0, 0.48)
        repair_gain = repair_gain * distressed.astype(float)
        s.hidden = np.clip(
            (1.0 - repair_gain[:, None]) * s.hidden + repair_gain[:, None] * repair_target,
            -2.0,
            2.0,
        )
        s.plasticity[distressed] = np.clip(s.plasticity[distressed] + 0.08 + 0.08 * z_support[distressed], 0.0, 1.0)
        s.commitment[distressed] = np.clip(s.commitment[distressed] - 0.06, 0.0, 1.0)
        s.energy[distressed] = np.clip(s.energy[distressed] + 0.015 * z_support[distressed], 0.0, 1.0)

    def medium_step(self, *, no_gradient: bool = False):
        s = self.state
        s.field_alignment = update_local_field_alignment(s.hidden)
        s.tissue_field = update_tissue_field(s.hidden, s.membrane, s.tissue_field, s.z_memory)
        s.oscillation_phase, s.oscillation_amplitude = update_oscillatory_coupling(
            s.hidden,
            s.membrane,
            s.conductance,
            s.tissue_field,
            s.stress,
            s.oscillation_phase,
            s.oscillation_amplitude,
        )
        s.morphogen_activator, s.morphogen_inhibitor = update_reaction_diffusion(
            s.hidden,
            s.tissue_field,
            s.oscillation_amplitude,
            s.morphogen_activator,
            s.morphogen_inhibitor,
        )
        s.highway_trace, s.highway_flux = update_stigmergic_highways(
            s.hidden,
            s.conductance,
            s.field_alignment,
            s.energy,
            s.highway_trace,
            s.highway_flux,
        )
        s.predictive_prediction, s.predictive_error, s.predictive_precision = update_predictive_coding(
            s.hidden,
            s.conductance,
            s.tissue_field,
            s.z_memory,
            s.predictive_prediction,
            s.predictive_error,
            s.predictive_precision,
        )
        if not no_gradient:
            s.z_memory = update_z_memory(s.hidden, s.z_alignment, s.z_memory)
        s.z_alignment = update_z_alignment(s.hidden, s.z_alignment, s.z_memory)
        if not no_gradient:
            s.conductance = update_conductance(s.conductance, s.membrane, s.stress)
            self._repair_parameter_drift()

    def slow_step(self, allow_growth: bool = True, *, no_gradient: bool = False):
        s = self.state
        if allow_growth:
            (
                s.energy,
                s.plasticity,
                s.commitment,
                s.growth_pressure,
                s.growth_cooldown,
                s.growth_activity,
                growth_diagnostics,
            ) = apply_regulated_growth(
                s.energy,
                s.stress,
                s.plasticity,
                s.commitment,
                s.field_alignment,
                s.z_alignment,
                s.predictive_error,
                s.growth_cooldown,
                s.growth_activity,
            )
            s.recent_growth_signal_mean = float(growth_diagnostics['growth_signal_mean'])
            s.recent_growth_event_fraction = float(growth_diagnostics['growth_event_fraction'])
            s.recent_growth_energy_transferred = float(growth_diagnostics['growth_energy_transferred'])
            s.recent_growth_repair_fraction = float(growth_diagnostics['growth_repair_fraction'])
            s.recent_growth_bottleneck_fraction = float(growth_diagnostics['growth_bottleneck_fraction'])
            s.recent_growth_decorative_fraction = float(growth_diagnostics['growth_decorative_fraction'])
            s.recent_structural_churn = float(growth_diagnostics['structural_churn'])

            active_growth = s.growth_activity > 0.12
            if np.any(active_growth):
                growth_support = np.clip(
                    0.55 * s.growth_activity[active_growth]
                    + 0.25 * s.growth_pressure[active_growth]
                    + 0.20 * s.predictive_error[active_growth],
                    0.0,
                    1.0,
                )
                s.stress[active_growth] = np.clip(
                    s.stress[active_growth] * (0.78 - 0.08 * growth_support) - 0.03 * growth_support,
                    0.0,
                    5.0,
                )
                s.energy[active_growth] = np.clip(
                    s.energy[active_growth] + 0.010 * growth_support,
                    0.0,
                    1.0,
                )
                s.field_alignment[active_growth] = np.clip(
                    s.field_alignment[active_growth] + 0.07 * growth_support,
                    0.0,
                    1.0,
                )
                s.z_alignment[active_growth] = np.clip(
                    s.z_alignment[active_growth] + 0.18 * growth_support * (s.z_memory[active_growth] - s.z_alignment[active_growth]),
                    -1.0,
                    1.0,
                )
        else:
            s.growth_pressure = np.zeros_like(s.growth_pressure)
            s.growth_cooldown = np.clip(s.growth_cooldown - 0.12, 0.0, 1.0)
            s.growth_activity = np.clip(s.growth_activity * 0.72, 0.0, 1.0)
            s.recent_growth_signal_mean = 0.0
            s.recent_growth_event_fraction = 0.0
            s.recent_growth_energy_transferred = 0.0
            s.recent_growth_repair_fraction = 0.0
            s.recent_growth_bottleneck_fraction = 0.0
            s.recent_growth_decorative_fraction = 0.0
            s.recent_structural_churn = float(np.mean(np.abs(s.growth_activity)))

        if not no_gradient:
            self._repair_port_regions()
        self._repair_distressed_regions(no_gradient=no_gradient)
        maturation_drive = 0.01 * (1.0 - np.clip(s.stress, 0.0, 1.0)) + 0.004 * np.clip(s.field_alignment, 0.0, 1.0)
        growth_offset = 0.006 * s.growth_pressure
        s.commitment = np.clip(s.commitment + maturation_drive - growth_offset, 0.0, 1.0)
        s.stages = assign_stages(s.plasticity, s.commitment, s.energy, s.stress)

    def step(
        self,
        do_fast: bool,
        do_medium: bool,
        do_slow: bool,
        noise_scale: float = 0.0,
        target_value: float = 0.75,
        allow_growth: bool = True,
        no_gradient: bool = False,
    ):
        if do_fast:
            self.fast_step(noise_scale, target_value)
        if do_medium:
            self.medium_step(no_gradient=no_gradient)
        if do_slow:
            self.slow_step(allow_growth=allow_growth, no_gradient=no_gradient)
        self.state.step_count += 1
        return self.state
