import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner, rollout_body
from morphobase.communication.oscillations import oscillation_phase_coherence


class OscillatoryCouplingProbeAssay(AssayRunner):
    def _region_bounds(self, num_cells: int) -> tuple[int, int]:
        width = max(4, num_cells // 6)
        start = max(1, num_cells // 4)
        stop = min(num_cells - 1, start + width)
        return start, stop

    def _window_phase_coherence(
        self,
        state_history: list,
        mask: np.ndarray,
        *,
        start_index: int,
        stop_index: int,
    ) -> float:
        tail = state_history[start_index:stop_index]
        coherences = [
            oscillation_phase_coherence(state.oscillation_phase[mask])
            for state in tail
        ]
        return float(np.mean(coherences))

    def _window_amplitude(
        self,
        state_history: list,
        mask: np.ndarray,
        *,
        start_index: int,
        stop_index: int,
    ) -> float:
        tail = state_history[start_index:stop_index]
        amplitudes = [
            float(np.mean(state.oscillation_amplitude[mask]))
            for state in tail
        ]
        return float(np.mean(amplitudes))

    def run(self, cfg):
        start, stop = self._region_bounds(cfg.body.num_cells)
        local_mask = np.zeros(cfg.body.num_cells, dtype=bool)
        local_mask[start:stop] = True
        distal_mask = ~local_mask
        perturb_start = cfg.runtime.total_steps // 3
        perturb_stop = min(cfg.runtime.total_steps, perturb_start + max(18, cfg.runtime.log_every * 2))

        def _drive_phase(step: int) -> float:
            return float(np.sin((step - perturb_start) * 0.55) * (np.pi / 2.5))

        def step_hook(step, body):
            if perturb_start <= step < perturb_stop:
                wave = np.sin((step - perturb_start) * 0.55)
                body.state.membrane[start:stop] = np.clip(body.state.membrane[start:stop] + 0.11 * wave, -1.0, 1.0)
                body.state.hidden[start:stop] = np.clip(body.state.hidden[start:stop] + 0.035 * wave, -2.0, 2.0)
                body.state.stress[start:stop] = np.clip(body.state.stress[start:stop] + 0.01 * abs(wave), 0.0, 5.0)

        def after_step_hook(step, body):
            if perturb_start <= step < perturb_stop:
                target_phase = _drive_phase(step)
                current_phase = body.state.oscillation_phase[start:stop]
                phase_delta = np.angle(np.exp(1j * (target_phase - current_phase)))
                body.state.oscillation_phase[start:stop] = current_phase + 0.45 * phase_delta
                body.state.oscillation_amplitude[start:stop] = np.clip(
                    body.state.oscillation_amplitude[start:stop] + 0.08,
                    0.0,
                    1.0,
                )
                body.state.oscillation_amplitude[distal_mask] = np.clip(
                    body.state.oscillation_amplitude[distal_mask] * 0.992,
                    0.0,
                    1.0,
                )

        baseline = rollout_body(cfg)
        perturbed = rollout_body(cfg, step_hook=step_hook, after_step_hook=after_step_hook)

        window_start = perturb_start + 1
        window_stop = perturb_stop + 1
        baseline_local_amp = self._window_amplitude(
            baseline['state_history'],
            local_mask,
            start_index=window_start,
            stop_index=window_stop,
        )
        baseline_distal_amp = self._window_amplitude(
            baseline['state_history'],
            distal_mask,
            start_index=window_start,
            stop_index=window_stop,
        )
        perturbed_local_amp = self._window_amplitude(
            perturbed['state_history'],
            local_mask,
            start_index=window_start,
            stop_index=window_stop,
        )
        perturbed_distal_amp = self._window_amplitude(
            perturbed['state_history'],
            distal_mask,
            start_index=window_start,
            stop_index=window_stop,
        )

        baseline_local_coherence = self._window_phase_coherence(
            baseline['state_history'],
            local_mask,
            start_index=window_start,
            stop_index=window_stop,
        )
        baseline_distal_coherence = self._window_phase_coherence(
            baseline['state_history'],
            distal_mask,
            start_index=window_start,
            stop_index=window_stop,
        )
        perturbed_local_coherence = self._window_phase_coherence(
            perturbed['state_history'],
            local_mask,
            start_index=window_start,
            stop_index=window_stop,
        )
        perturbed_distal_coherence = self._window_phase_coherence(
            perturbed['state_history'],
            distal_mask,
            start_index=window_start,
            stop_index=window_stop,
        )

        local_amp_gain = perturbed_local_amp - baseline_local_amp
        distal_amp_gain = perturbed_distal_amp - baseline_distal_amp
        local_coherence_gain = perturbed_local_coherence - baseline_local_coherence
        distal_coherence_gain = perturbed_distal_coherence - baseline_distal_coherence
        localization_ratio = local_amp_gain / max(abs(distal_amp_gain), 1e-8)
        coherence_advantage = local_coherence_gain - distal_coherence_gain

        perturbed_metrics = perturbed['final_metrics'].copy()
        perturbed_metrics.update(
            {
                'baseline_mean_oscillation_amplitude': baseline['final_metrics']['mean_oscillation_amplitude'],
                'baseline_oscillation_phase_coherence': baseline['final_metrics']['oscillation_phase_coherence'],
                'oscillation_local_amplitude_gain': float(local_amp_gain),
                'oscillation_distal_amplitude_gain': float(distal_amp_gain),
                'oscillation_localization_ratio': float(localization_ratio),
                'oscillation_local_phase_coherence_gain': float(local_coherence_gain),
                'oscillation_distal_phase_coherence_gain': float(distal_coherence_gain),
                'oscillation_coherence_advantage': float(coherence_advantage),
                'oscillation_probe_region_width': float(stop - start),
                'oscillation_bounded': float(
                    np.max(np.abs(perturbed['body'].state.oscillation_phase)) <= np.pi + 1e-8
                    and np.max(perturbed['body'].state.oscillation_amplitude) <= 1.0 + 1e-8
                ),
            }
        )

        notes = (
            'Oscillatory coupling probe completed with a localized oscillatory drive. '
            f"Local amplitude gain was {local_amp_gain:.4f} versus distal {distal_amp_gain:.4f}, "
            f'with coherence advantage {coherence_advantage:.4f}.'
        )
        return AssayResult(
            history=perturbed['history'],
            final_metrics=perturbed_metrics,
            notes=notes,
        )
