import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner, rollout_body


class StigmergicHighwayProbeAssay(AssayRunner):
    def _region_bounds(self, num_cells: int) -> tuple[int, int]:
        width = max(5, num_cells // 5)
        start = max(1, num_cells // 4)
        stop = min(num_cells - 1, start + width)
        return start, stop

    def _window_trace(self, state_history: list, *, start_index: int, stop_index: int) -> np.ndarray:
        window = state_history[start_index:stop_index]
        return np.mean([state.highway_trace for state in window], axis=0)

    def run(self, cfg):
        start, stop = self._region_bounds(cfg.body.num_cells)
        local_mask = np.zeros(cfg.body.num_cells, dtype=bool)
        local_mask[start:stop] = True
        distal_mask = ~local_mask
        deposit_start = cfg.runtime.total_steps // 4
        deposit_stop = min(cfg.runtime.total_steps, deposit_start + max(18, cfg.runtime.log_every * 3))
        persistence_start = min(cfg.runtime.total_steps, deposit_stop + 1)
        persistence_stop = min(cfg.runtime.total_steps, persistence_start + max(10, cfg.runtime.log_every * 2))

        def step_hook(step, body):
            if deposit_start <= step < deposit_stop:
                body.state.hidden[start:stop] = np.clip(body.state.hidden[start:stop] + 0.04, -2.0, 2.0)
                body.state.membrane[start:stop] = np.clip(body.state.membrane[start:stop] + 0.05, -1.0, 1.0)
                body.state.field_alignment[start:stop] = np.clip(body.state.field_alignment[start:stop] + 0.04, 0.0, 1.0)
                body.state.conductance[start:stop, start:stop] = np.clip(
                    body.state.conductance[start:stop, start:stop] + 0.06,
                    0.0,
                    2.0,
                )
                diagonal = np.diag_indices_from(body.state.conductance)
                body.state.conductance[diagonal] = 1.0

        def after_step_hook(step, body):
            if deposit_start <= step < deposit_stop:
                body.state.highway_trace[start:stop] = np.clip(
                    body.state.highway_trace[start:stop] + 0.05,
                    0.0,
                    1.0,
                )
                body.state.highway_flux[start:stop] = np.clip(
                    body.state.highway_flux[start:stop] + 0.04,
                    0.0,
                    1.0,
                )

        baseline = rollout_body(cfg)
        perturbed = rollout_body(cfg, step_hook=step_hook, after_step_hook=after_step_hook)

        drive_start = deposit_start + 1
        drive_stop = deposit_stop + 1
        baseline_drive_trace = self._window_trace(baseline['state_history'], start_index=drive_start, stop_index=drive_stop)
        perturbed_drive_trace = self._window_trace(perturbed['state_history'], start_index=drive_start, stop_index=drive_stop)
        drive_delta = perturbed_drive_trace - baseline_drive_trace

        if persistence_stop > persistence_start:
            baseline_persistence = self._window_trace(
                baseline['state_history'],
                start_index=persistence_start + 1,
                stop_index=persistence_stop + 1,
            )
            perturbed_persistence = self._window_trace(
                perturbed['state_history'],
                start_index=persistence_start + 1,
                stop_index=persistence_stop + 1,
            )
        else:
            baseline_persistence = baseline_drive_trace
            perturbed_persistence = perturbed_drive_trace
        persistence_delta = perturbed_persistence - baseline_persistence

        local_response = float(np.mean(np.abs(drive_delta[local_mask])))
        distal_response = float(np.mean(np.abs(drive_delta[distal_mask])))
        localization_ratio = local_response / max(distal_response, 1e-8)
        persistence_local = float(np.mean(np.abs(persistence_delta[local_mask])))
        persistence_retention = persistence_local / max(local_response, 1e-8)

        perturbed_metrics = perturbed['final_metrics'].copy()
        baseline_metrics = baseline['final_metrics']
        perturbed_metrics.update(
            {
                'baseline_stigmergic_highway_strength': baseline_metrics['stigmergic_highway_strength'],
                'baseline_stigmergic_tissue_coupling': baseline_metrics['stigmergic_tissue_coupling'],
                'stigmergic_local_response': local_response,
                'stigmergic_distal_response': distal_response,
                'stigmergic_localization_ratio': float(localization_ratio),
                'stigmergic_persistence_local': persistence_local,
                'stigmergic_persistence_retention': float(persistence_retention),
                'stigmergic_highway_gain': float(
                    perturbed_metrics['stigmergic_highway_strength'] - baseline_metrics['stigmergic_highway_strength']
                ),
                'stigmergic_tissue_coupling_gain': float(
                    perturbed_metrics['stigmergic_tissue_coupling'] - baseline_metrics['stigmergic_tissue_coupling']
                ),
                'stigmergic_region_width': float(stop - start),
                'stigmergic_bounded': float(
                    np.max(perturbed['body'].state.highway_trace) <= 1.0 + 1e-8
                    and np.min(perturbed['body'].state.highway_trace) >= -1e-8
                    and np.max(perturbed['body'].state.highway_flux) <= 1.0 + 1e-8
                    and np.min(perturbed['body'].state.highway_flux) >= -1e-8
                ),
            }
        )

        notes = (
            'Stigmergic highway probe completed with a localized repeated-traffic deposit. '
            f"Local highway response was {local_response:.4f} versus distal {distal_response:.4f}, "
            f'with persistence retention {persistence_retention:.2f}.'
        )
        return AssayResult(
            history=perturbed['history'],
            final_metrics=perturbed_metrics,
            notes=notes,
        )
