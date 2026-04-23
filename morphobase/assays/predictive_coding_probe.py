import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner, rollout_body


class PredictiveCodingProbeAssay(AssayRunner):
    def _region_bounds(self, num_cells: int) -> tuple[int, int]:
        width = max(5, num_cells // 5)
        start = max(1, num_cells // 3)
        stop = min(num_cells - 1, start + width)
        return start, stop

    def _window_mean(self, state_history: list, attr: str, mask: np.ndarray, *, start_index: int, stop_index: int) -> float:
        window = state_history[start_index:stop_index]
        return float(np.mean([getattr(state, attr)[mask].mean() for state in window]))

    def run(self, cfg):
        start, stop = self._region_bounds(cfg.body.num_cells)
        local_mask = np.zeros(cfg.body.num_cells, dtype=bool)
        local_mask[start:stop] = True
        distal_mask = ~local_mask
        mismatch_start = cfg.runtime.total_steps // 4
        mismatch_stop = min(cfg.runtime.total_steps, mismatch_start + max(16, cfg.runtime.log_every * 2))
        recovery_start = mismatch_stop + 1
        recovery_stop = min(cfg.runtime.total_steps, recovery_start + max(12, cfg.runtime.log_every * 2))

        def step_hook(step, body):
            if mismatch_start <= step < mismatch_stop:
                body.state.hidden[start:stop] = np.clip(body.state.hidden[start:stop] + 0.12, -2.0, 2.0)
                body.state.membrane[start:stop] = np.clip(body.state.membrane[start:stop] - 0.08, -1.0, 1.0)
                body.state.tissue_field[start:stop] = np.clip(body.state.tissue_field[start:stop] - 0.04, -1.0, 1.0)

        def after_step_hook(step, body):
            if recovery_start <= step < recovery_stop:
                local_signal = np.mean(np.tanh(body.state.hidden[start:stop]), axis=1)
                body.state.predictive_prediction[start:stop] = np.clip(
                    0.70 * body.state.predictive_prediction[start:stop] + 0.30 * local_signal,
                    -1.0,
                    1.0,
                )
                corrected_error = np.abs(local_signal - body.state.predictive_prediction[start:stop])
                body.state.predictive_error[start:stop] = np.clip(
                    0.55 * body.state.predictive_error[start:stop] + 0.45 * corrected_error,
                    0.0,
                    1.0,
                )
                body.state.predictive_precision[start:stop] = np.clip(
                    body.state.predictive_precision[start:stop] + 0.05,
                    0.0,
                    1.0,
                )

        baseline = rollout_body(cfg)
        perturbed = rollout_body(cfg, step_hook=step_hook, after_step_hook=after_step_hook)

        mismatch_window = (mismatch_start + 1, mismatch_stop + 1)
        recovery_window = (recovery_start + 1, recovery_stop + 1) if recovery_stop > recovery_start else mismatch_window

        baseline_local_error = self._window_mean(
            baseline['state_history'],
            'predictive_error',
            local_mask,
            start_index=mismatch_window[0],
            stop_index=mismatch_window[1],
        )
        baseline_distal_error = self._window_mean(
            baseline['state_history'],
            'predictive_error',
            distal_mask,
            start_index=mismatch_window[0],
            stop_index=mismatch_window[1],
        )
        perturbed_local_error = self._window_mean(
            perturbed['state_history'],
            'predictive_error',
            local_mask,
            start_index=mismatch_window[0],
            stop_index=mismatch_window[1],
        )
        perturbed_distal_error = self._window_mean(
            perturbed['state_history'],
            'predictive_error',
            distal_mask,
            start_index=mismatch_window[0],
            stop_index=mismatch_window[1],
        )

        local_error_response = perturbed_local_error - baseline_local_error
        distal_error_response = perturbed_distal_error - baseline_distal_error
        localization_ratio = local_error_response / max(abs(distal_error_response), 1e-8)

        perturbed_local_recovery_error = self._window_mean(
            perturbed['state_history'],
            'predictive_error',
            local_mask,
            start_index=recovery_window[0],
            stop_index=recovery_window[1],
        )
        perturbed_local_recovery_precision = self._window_mean(
            perturbed['state_history'],
            'predictive_precision',
            local_mask,
            start_index=recovery_window[0],
            stop_index=recovery_window[1],
        )
        baseline_local_recovery_precision = self._window_mean(
            baseline['state_history'],
            'predictive_precision',
            local_mask,
            start_index=recovery_window[0],
            stop_index=recovery_window[1],
        )

        error_recovery_gain = perturbed_local_error - perturbed_local_recovery_error
        precision_gain = perturbed_local_recovery_precision - baseline_local_recovery_precision

        perturbed_metrics = perturbed['final_metrics'].copy()
        baseline_metrics = baseline['final_metrics']
        perturbed_metrics.update(
            {
                'baseline_mean_predictive_error': baseline_metrics['mean_predictive_error'],
                'baseline_mean_predictive_precision': baseline_metrics['mean_predictive_precision'],
                'predictive_local_error_response': float(local_error_response),
                'predictive_distal_error_response': float(distal_error_response),
                'predictive_localization_ratio': float(localization_ratio),
                'predictive_error_recovery_gain': float(error_recovery_gain),
                'predictive_precision_gain': float(precision_gain),
                'predictive_probe_region_width': float(stop - start),
                'predictive_coding_bounded': float(
                    np.max(np.abs(perturbed['body'].state.predictive_prediction)) <= 1.0 + 1e-8
                    and np.max(perturbed['body'].state.predictive_error) <= 1.0 + 1e-8
                    and np.min(perturbed['body'].state.predictive_error) >= -1e-8
                    and np.max(perturbed['body'].state.predictive_precision) <= 1.0 + 1e-8
                    and np.min(perturbed['body'].state.predictive_precision) >= -1e-8
                ),
            }
        )

        notes = (
            'Predictive coding probe completed with a localized mismatch intervention. '
            f"Local predictive-error response was {local_error_response:.4f} versus distal {distal_error_response:.4f}, "
            f'with recovery gain {error_recovery_gain:.4f}.'
        )
        return AssayResult(history=perturbed['history'], final_metrics=perturbed_metrics, notes=notes)
