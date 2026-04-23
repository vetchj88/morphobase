from dataclasses import dataclass

import numpy as np

from morphobase.types import Stage


@dataclass(slots=True)
class OrganismState:
    hidden: np.ndarray
    membrane: np.ndarray
    plasticity: np.ndarray
    energy: np.ndarray
    stress: np.ndarray
    role_logits: np.ndarray
    commitment: np.ndarray
    field_alignment: np.ndarray
    tissue_field: np.ndarray
    oscillation_phase: np.ndarray
    oscillation_amplitude: np.ndarray
    morphogen_activator: np.ndarray
    morphogen_inhibitor: np.ndarray
    highway_trace: np.ndarray
    highway_flux: np.ndarray
    predictive_prediction: np.ndarray
    predictive_error: np.ndarray
    predictive_precision: np.ndarray
    growth_cooldown: np.ndarray
    growth_activity: np.ndarray
    z_alignment: np.ndarray
    z_memory: np.ndarray
    alive: np.ndarray
    conductance: np.ndarray
    stages: np.ndarray
    growth_pressure: np.ndarray
    recent_growth_signal_mean: float = 0.0
    recent_growth_event_fraction: float = 0.0
    recent_growth_energy_transferred: float = 0.0
    recent_growth_repair_fraction: float = 0.0
    recent_growth_bottleneck_fraction: float = 0.0
    recent_growth_decorative_fraction: float = 0.0
    recent_structural_churn: float = 0.0
    step_count: int = 0

    @classmethod
    def synthetic(
        cls,
        num_cells: int,
        hidden_dim: int,
        energy_init: float,
        stress_init: float,
        plasticity_init: float,
        z_alignment_init: float,
    ):
        return cls(
            hidden=np.zeros((num_cells, hidden_dim), dtype=float),
            membrane=np.zeros(num_cells, dtype=float),
            plasticity=np.full(num_cells, plasticity_init, dtype=float),
            energy=np.full(num_cells, energy_init, dtype=float),
            stress=np.full(num_cells, stress_init, dtype=float),
            role_logits=np.zeros((num_cells, 4), dtype=float),
            commitment=np.zeros(num_cells, dtype=float),
            field_alignment=np.zeros(num_cells, dtype=float),
            tissue_field=np.zeros(num_cells, dtype=float),
            oscillation_phase=np.linspace(-np.pi, np.pi, num_cells, dtype=float),
            oscillation_amplitude=np.zeros(num_cells, dtype=float),
            morphogen_activator=np.zeros(num_cells, dtype=float),
            morphogen_inhibitor=np.zeros(num_cells, dtype=float),
            highway_trace=np.zeros(num_cells, dtype=float),
            highway_flux=np.zeros(num_cells, dtype=float),
            predictive_prediction=np.zeros(num_cells, dtype=float),
            predictive_error=np.zeros(num_cells, dtype=float),
            predictive_precision=np.zeros(num_cells, dtype=float),
            growth_cooldown=np.zeros(num_cells, dtype=float),
            growth_activity=np.zeros(num_cells, dtype=float),
            z_alignment=np.full(num_cells, z_alignment_init, dtype=float),
            z_memory=np.full(num_cells, z_alignment_init, dtype=float),
            alive=np.ones(num_cells, dtype=bool),
            conductance=np.eye(num_cells, dtype=float),
            stages=np.array([Stage.SEED.value] * num_cells, dtype=object),
            growth_pressure=np.zeros(num_cells, dtype=float),
        )

    def copy(self) -> 'OrganismState':
        return OrganismState(
            hidden=self.hidden.copy(),
            membrane=self.membrane.copy(),
            plasticity=self.plasticity.copy(),
            energy=self.energy.copy(),
            stress=self.stress.copy(),
            role_logits=self.role_logits.copy(),
            commitment=self.commitment.copy(),
            field_alignment=self.field_alignment.copy(),
            tissue_field=self.tissue_field.copy(),
            oscillation_phase=self.oscillation_phase.copy(),
            oscillation_amplitude=self.oscillation_amplitude.copy(),
            morphogen_activator=self.morphogen_activator.copy(),
            morphogen_inhibitor=self.morphogen_inhibitor.copy(),
            highway_trace=self.highway_trace.copy(),
            highway_flux=self.highway_flux.copy(),
            predictive_prediction=self.predictive_prediction.copy(),
            predictive_error=self.predictive_error.copy(),
            predictive_precision=self.predictive_precision.copy(),
            growth_cooldown=self.growth_cooldown.copy(),
            growth_activity=self.growth_activity.copy(),
            z_alignment=self.z_alignment.copy(),
            z_memory=self.z_memory.copy(),
            alive=self.alive.copy(),
            conductance=self.conductance.copy(),
            stages=self.stages.copy(),
            growth_pressure=self.growth_pressure.copy(),
            recent_growth_signal_mean=self.recent_growth_signal_mean,
            recent_growth_event_fraction=self.recent_growth_event_fraction,
            recent_growth_energy_transferred=self.recent_growth_energy_transferred,
            recent_growth_repair_fraction=self.recent_growth_repair_fraction,
            recent_growth_bottleneck_fraction=self.recent_growth_bottleneck_fraction,
            recent_growth_decorative_fraction=self.recent_growth_decorative_fraction,
            recent_structural_churn=self.recent_structural_churn,
            step_count=self.step_count,
        )
