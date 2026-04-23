import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner, rollout_body


class ReactionDiffusionProbeAssay(AssayRunner):
    def _region_bounds(self, num_cells: int) -> tuple[int, int]:
        width = max(5, num_cells // 5)
        start = max(1, num_cells // 3)
        stop = min(num_cells - 1, start + width)
        return start, stop

    def _window_pattern(self, state_history: list, *, start_index: int, stop_index: int) -> np.ndarray:
        window = state_history[start_index:stop_index]
        return np.mean(
            [state.morphogen_activator - state.morphogen_inhibitor for state in window],
            axis=0,
        )

    def _window_pattern_strength(self, state_history: list, *, start_index: int, stop_index: int) -> float:
        pattern = self._window_pattern(state_history, start_index=start_index, stop_index=stop_index)
        return float(np.mean(np.abs(pattern - np.mean(pattern))))

    def run(self, cfg):
        start, stop = self._region_bounds(cfg.body.num_cells)
        local_mask = np.zeros(cfg.body.num_cells, dtype=bool)
        local_mask[start:stop] = True
        distal_mask = ~local_mask
        perturb_start = cfg.runtime.total_steps // 4
        perturb_stop = min(cfg.runtime.total_steps, perturb_start + max(20, cfg.runtime.log_every * 3))

        def step_hook(step, body):
            if perturb_start <= step < perturb_stop:
                body.state.tissue_field[start:stop] = np.clip(body.state.tissue_field[start:stop] + 0.05, -1.0, 1.0)
                body.state.hidden[start:stop] = np.clip(body.state.hidden[start:stop] + 0.03, -2.0, 2.0)
                body.state.membrane[start:stop] = np.clip(body.state.membrane[start:stop] + 0.04, -1.0, 1.0)
                body.state.oscillation_amplitude[start:stop] = np.clip(
                    body.state.oscillation_amplitude[start:stop] + 0.03,
                    0.0,
                    1.0,
                )

        baseline = rollout_body(cfg)
        perturbed = rollout_body(cfg, step_hook=step_hook)

        window_start = perturb_start + 1
        window_stop = perturb_stop + 1
        baseline_pattern = self._window_pattern(baseline['state_history'], start_index=window_start, stop_index=window_stop)
        perturbed_pattern = self._window_pattern(perturbed['state_history'], start_index=window_start, stop_index=window_stop)
        delta_pattern = perturbed_pattern - baseline_pattern
        baseline_window_strength = self._window_pattern_strength(
            baseline['state_history'],
            start_index=window_start,
            stop_index=window_stop,
        )
        perturbed_window_strength = self._window_pattern_strength(
            perturbed['state_history'],
            start_index=window_start,
            stop_index=window_stop,
        )

        local_response = float(np.mean(np.abs(delta_pattern[local_mask])))
        distal_response = float(np.mean(np.abs(delta_pattern[distal_mask])))
        localization_ratio = local_response / max(distal_response, 1e-8)
        local_sign = float(np.mean(delta_pattern[local_mask]))
        distal_sign = float(np.mean(delta_pattern[distal_mask]))

        perturbed_metrics = perturbed['final_metrics'].copy()
        baseline_metrics = baseline['final_metrics']
        perturbed_metrics.update(
            {
                'baseline_reaction_diffusion_pattern_strength': baseline_metrics['reaction_diffusion_pattern_strength'],
                'baseline_reaction_diffusion_tissue_coupling': baseline_metrics['reaction_diffusion_tissue_coupling'],
                'reaction_diffusion_local_response': local_response,
                'reaction_diffusion_distal_response': distal_response,
                'reaction_diffusion_localization_ratio': float(localization_ratio),
                'reaction_diffusion_pattern_gain': float(perturbed_window_strength - baseline_window_strength),
                'reaction_diffusion_tissue_coupling_gain': float(
                    perturbed_metrics['reaction_diffusion_tissue_coupling']
                    - baseline_metrics['reaction_diffusion_tissue_coupling']
                ),
                'reaction_diffusion_window_pattern_strength': float(perturbed_window_strength),
                'reaction_diffusion_local_sign': local_sign,
                'reaction_diffusion_distal_sign': distal_sign,
                'reaction_diffusion_region_width': float(stop - start),
                'reaction_diffusion_bounded': float(
                    np.max(perturbed['body'].state.morphogen_activator) <= 1.0 + 1e-8
                    and np.min(perturbed['body'].state.morphogen_activator) >= -1e-8
                    and np.max(perturbed['body'].state.morphogen_inhibitor) <= 1.0 + 1e-8
                    and np.min(perturbed['body'].state.morphogen_inhibitor) >= -1e-8
                ),
            }
        )

        notes = (
            'Reaction-diffusion probe completed with a localized morphogen bias. '
            f"Local pattern response was {local_response:.4f} versus distal {distal_response:.4f}, "
            f'for a localization ratio of {localization_ratio:.2f}.'
        )
        return AssayResult(
            history=perturbed['history'],
            final_metrics=perturbed_metrics,
            notes=notes,
        )
