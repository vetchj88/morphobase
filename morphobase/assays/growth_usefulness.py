import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner
from morphobase.diagnostics.metrics import lightcone_proxy, summarize_state
from morphobase.organism.body import Body
from morphobase.organism.scheduler import Scheduler
from morphobase.organism.state import OrganismState
from morphobase.pathology.lesions import lesion_cells


class GrowthUsefulnessAssay(AssayRunner):
    def _rollout(self, cfg, allow_growth: bool):
        state = OrganismState.synthetic(
            cfg.body.num_cells,
            cfg.body.hidden_dim,
            cfg.body.energy_init,
            cfg.body.stress_init,
            cfg.body.plasticity_init,
            cfg.body.z_alignment_init,
        )
        body = Body(state)
        scheduler = Scheduler()
        history = []
        z_history = [state.z_alignment.copy()]
        state_history = [state.copy()]
        lesion_start = max(1, cfg.body.num_cells // 3)
        lesion_stop = min(cfg.body.num_cells, lesion_start + max(2, cfg.body.num_cells // 5))
        lesion_slice = slice(lesion_start, lesion_stop)
        lesion_step = cfg.runtime.total_steps // 3
        bottleneck_stop = min(cfg.runtime.total_steps, lesion_step + cfg.runtime.total_steps // 4)

        for step in range(cfg.runtime.total_steps):
            if step == lesion_step:
                body.state.hidden = lesion_cells(body.state.hidden, lesion_start, lesion_stop)
                body.state.energy[lesion_start:lesion_stop] = body.state.energy[lesion_start:lesion_stop] * 0.4
                body.state.stress[lesion_start:lesion_stop] = body.state.stress[lesion_start:lesion_stop] + 0.6
                body.state.field_alignment[lesion_start:lesion_stop] = (
                    body.state.field_alignment[lesion_start:lesion_stop] * 0.55
                )
                body.state.z_alignment[lesion_start:lesion_stop] = (
                    body.state.z_alignment[lesion_start:lesion_stop] * 0.65
                )
                body.state.conductance[lesion_start:lesion_stop, :] *= 0.40
                body.state.conductance[:, lesion_start:lesion_stop] *= 0.40
                diagonal = np.diag_indices_from(body.state.conductance)
                body.state.conductance[diagonal] = 1.0
            elif lesion_step < step < bottleneck_stop and step % max(cfg.runtime.log_every // 2, 2) == 0:
                body.state.energy[lesion_start:lesion_stop] = (
                    body.state.energy[lesion_start:lesion_stop] * 0.97
                )
                body.state.stress[lesion_start:lesion_stop] = (
                    body.state.stress[lesion_start:lesion_stop] + 0.08
                )
                body.state.field_alignment[lesion_start:lesion_stop] = (
                    body.state.field_alignment[lesion_start:lesion_stop] * 0.96
                )

            due = scheduler.due(step)
            body.step(
                due.fast,
                due.medium,
                due.slow,
                cfg.assay.noise_scale,
                cfg.assay.target_value,
                allow_growth=allow_growth,
            )
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            if step % cfg.runtime.log_every == 0 or step == cfg.runtime.total_steps - 1:
                history.append(summarize_state(body.state, z_history=z_history))

        final_metrics = history[-1].copy()
        final_metrics['lightcone_proxy'] = lightcone_proxy(state_history)
        final_metrics['lesion_region_mean_energy'] = float(body.state.energy[lesion_slice].mean())
        final_metrics['lesion_region_mean_stress'] = float(body.state.stress[lesion_slice].mean())
        final_metrics['lesion_region_mean_field_alignment'] = float(body.state.field_alignment[lesion_slice].mean())
        final_metrics['lesion_region_mean_z_alignment'] = float(body.state.z_alignment[lesion_slice].mean())
        return history, final_metrics

    def run(self, cfg):
        history, grown_metrics = self._rollout(cfg, allow_growth=True)
        baseline_history, baseline_metrics = self._rollout(cfg, allow_growth=False)

        grown_metrics['baseline_mean_energy'] = baseline_metrics['mean_energy']
        grown_metrics['baseline_mean_z_alignment'] = baseline_metrics['mean_z_alignment']
        grown_metrics['baseline_lesion_region_mean_energy'] = baseline_metrics['lesion_region_mean_energy']
        grown_metrics['baseline_lesion_region_mean_stress'] = baseline_metrics['lesion_region_mean_stress']
        grown_metrics['baseline_lesion_region_mean_field_alignment'] = baseline_metrics['lesion_region_mean_field_alignment']
        grown_metrics['baseline_lesion_region_mean_z_alignment'] = baseline_metrics['lesion_region_mean_z_alignment']
        grown_metrics['energy_advantage'] = grown_metrics['mean_energy'] - baseline_metrics['mean_energy']
        grown_metrics['z_alignment_advantage'] = grown_metrics['mean_z_alignment'] - baseline_metrics['mean_z_alignment']
        grown_metrics['lesion_energy_advantage'] = (
            grown_metrics['lesion_region_mean_energy'] - baseline_metrics['lesion_region_mean_energy']
        )
        grown_metrics['lesion_stress_advantage'] = (
            baseline_metrics['lesion_region_mean_stress'] - grown_metrics['lesion_region_mean_stress']
        )
        grown_metrics['lesion_field_advantage'] = (
            grown_metrics['lesion_region_mean_field_alignment'] - baseline_metrics['lesion_region_mean_field_alignment']
        )
        grown_metrics['lesion_z_advantage'] = (
            grown_metrics['lesion_region_mean_z_alignment'] - baseline_metrics['lesion_region_mean_z_alignment']
        )
        utility_gain = (
            0.35 * grown_metrics['lesion_energy_advantage']
            + 0.30 * grown_metrics['lesion_stress_advantage']
            + 0.20 * grown_metrics['lesion_field_advantage']
            + 0.15 * grown_metrics['lesion_z_advantage']
        )
        cumulative_growth_energy = float(sum(item.get('recent_growth_energy_transferred', 0.0) for item in history))
        peak_growth_event_fraction = float(max(item.get('recent_growth_event_fraction', 0.0) for item in history))
        mean_growth_repair_fraction = float(
            sum(item.get('recent_growth_repair_fraction', 0.0) for item in history) / max(len(history), 1)
        )
        mean_growth_decorative_fraction = float(
            sum(item.get('recent_growth_decorative_fraction', 0.0) for item in history) / max(len(history), 1)
        )
        late_history = history[max(len(history) * 2 // 3, 0):]
        late_growth_event_fraction_mean = float(
            sum(item.get('recent_growth_event_fraction', 0.0) for item in late_history) / max(len(late_history), 1)
        )
        baseline_late_history = baseline_history[max(len(baseline_history) * 2 // 3, 0):]
        baseline_late_growth_event_fraction_mean = float(
            sum(item.get('recent_growth_event_fraction', 0.0) for item in baseline_late_history)
            / max(len(baseline_late_history), 1)
        )
        growth_cost = max(cumulative_growth_energy, 1e-6)
        grown_metrics['growth_utility_gain'] = utility_gain
        grown_metrics['growth_efficiency_advantage'] = utility_gain / growth_cost
        grown_metrics['cumulative_growth_energy_transferred'] = cumulative_growth_energy
        grown_metrics['peak_growth_event_fraction'] = peak_growth_event_fraction
        grown_metrics['mean_growth_repair_fraction'] = mean_growth_repair_fraction
        grown_metrics['mean_growth_decorative_fraction'] = mean_growth_decorative_fraction
        grown_metrics['late_growth_event_fraction_mean'] = late_growth_event_fraction_mean
        grown_metrics['baseline_late_growth_event_fraction_mean'] = baseline_late_growth_event_fraction_mean
        grown_metrics['decorative_growth_penalty'] = (
            mean_growth_decorative_fraction * (1.0 + late_growth_event_fraction_mean)
        )

        notes = (
            'Growth usefulness assay completed with mid-run lesion and selective redistribution. '
            f"Lesion utility gain vs no-growth control: {grown_metrics['growth_utility_gain']:.4f}. "
            f"Efficiency advantage: {grown_metrics['growth_efficiency_advantage']:.4f}. "
            f"Decorative growth penalty: {grown_metrics['decorative_growth_penalty']:.4f}."
        )
        return AssayResult(history=history, final_metrics=grown_metrics, notes=notes)
