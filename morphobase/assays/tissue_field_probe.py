import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner, bias_z_field, rollout_body


class TissueFieldProbeAssay(AssayRunner):
    def _region_bounds(self, num_cells: int) -> tuple[int, int]:
        width = max(4, num_cells // 6)
        start = max(1, (num_cells - width) // 2)
        stop = min(num_cells - 1, start + width)
        return start, stop

    def _window_mean_tissue_field(self, state_history: list, window: int = 8) -> np.ndarray:
        tail = state_history[-window:]
        return np.mean([state.tissue_field for state in tail], axis=0)

    def run(self, cfg):
        start, stop = self._region_bounds(cfg.body.num_cells)
        perturb_start = cfg.runtime.total_steps // 3
        perturb_stop = min(cfg.runtime.total_steps, perturb_start + max(6, cfg.runtime.log_every))

        def step_hook(step, body):
            if perturb_start <= step < perturb_stop:
                body.state.hidden[start:stop] = np.clip(body.state.hidden[start:stop] + 0.08, -2.0, 2.0)
                body.state.membrane[start:stop] = np.clip(body.state.membrane[start:stop] + 0.05, -1.0, 1.0)
                body.state.energy[start:stop] = np.clip(body.state.energy[start:stop] * 0.996, 0.0, 1.0)
                body.state.stress[start:stop] = np.clip(body.state.stress[start:stop] + 0.03, 0.0, 5.0)
                bias_z_field(body, start, stop, 0.03)

        baseline = rollout_body(cfg)
        perturbed = rollout_body(cfg, step_hook=step_hook)

        baseline_tissue = self._window_mean_tissue_field(baseline['state_history'])
        perturbed_tissue = self._window_mean_tissue_field(perturbed['state_history'])
        local_mask = np.zeros(cfg.body.num_cells, dtype=bool)
        local_mask[start:stop] = True
        distal_mask = ~local_mask

        local_response = float(np.mean(np.abs(perturbed_tissue[local_mask] - baseline_tissue[local_mask])))
        distal_response = float(np.mean(np.abs(perturbed_tissue[distal_mask] - baseline_tissue[distal_mask])))
        localization_ratio = local_response / max(distal_response, 1e-8)
        perturbed_metrics = perturbed['final_metrics'].copy()
        baseline_metrics = baseline['final_metrics']
        perturbed_metrics.update(
            {
                'baseline_mean_tissue_field': baseline_metrics['mean_tissue_field'],
                'baseline_tissue_field_variance': baseline_metrics['tissue_field_variance'],
                'baseline_tissue_field_regionality': baseline_metrics['tissue_field_regionality'],
                'baseline_tissue_field_z_coupling': baseline_metrics['tissue_field_z_coupling'],
                'tissue_field_local_response': local_response,
                'tissue_field_distal_response': distal_response,
                'tissue_field_localization_ratio': float(localization_ratio),
                'tissue_field_regionality_gain': float(
                    perturbed_metrics['tissue_field_regionality'] - baseline_metrics['tissue_field_regionality']
                ),
                'tissue_field_z_coupling_gain': float(
                    perturbed_metrics['tissue_field_z_coupling'] - baseline_metrics['tissue_field_z_coupling']
                ),
                'tissue_field_peak_magnitude': float(np.max(np.abs(perturbed_tissue))),
                'tissue_field_probe_region_width': float(stop - start),
                'tissue_field_bounded': float(np.max(np.abs(perturbed_tissue)) <= 1.0 + 1e-8),
            }
        )

        notes = (
            'Tissue field probe completed with a sustained regional perturbation. '
            f"Local tissue-field response was {local_response:.4f} versus distal {distal_response:.4f}, "
            f'for a localization ratio of {localization_ratio:.2f}.'
        )
        return AssayResult(
            history=perturbed['history'],
            final_metrics=perturbed_metrics,
            notes=notes,
        )
