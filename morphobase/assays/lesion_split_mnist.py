from __future__ import annotations

import numpy as np

from morphobase.assays.common import AssayResult, disrupt_port_region, lesion_segment
from morphobase.assays.split_mnist import SplitMNISTAssay
from morphobase.ports.mnist_port import MNISTPort


class LesionSplitMNISTAssay(SplitMNISTAssay):
    METRIC_PREFIX = "lesion_split_mnist"
    TASK_LABEL = "Lesion-split-MNIST"
    ROW_LESION_START = 10
    ROW_LESION_STOP = 18
    SETTLE_STEPS = 30

    def __init__(self) -> None:
        super().__init__()
        self.challenge_variant = "growth_probe"
        self._current_setpoint = 0.0

    @staticmethod
    def _latent_setpoint(image: np.ndarray) -> float:
        image_signal = np.asarray(image, dtype=float)
        top = float(np.mean(image_signal[: image_signal.shape[0] // 2]))
        bottom = float(np.mean(image_signal[image_signal.shape[0] // 2 :]))
        left = float(np.mean(image_signal[:, : image_signal.shape[1] // 2]))
        right = float(np.mean(image_signal[:, image_signal.shape[1] // 2 :]))
        contrast = float(np.std(image_signal))
        return float(np.clip(np.tanh(2.2 * (top - bottom) + 1.6 * (left - right) + 1.4 * contrast), -1.0, 1.0))

    def _apply_task_context(self, body, port: MNISTPort, step_index: int) -> None:
        row_index = max(port.row_count - 1, 0)
        output_mask = port.support_mask("output")
        if row_index <= self.ROW_LESION_START:
            body.state.z_memory[output_mask] = np.clip(
                0.80 * body.state.z_memory[output_mask] + 0.20 * self._current_setpoint,
                -1.0,
                1.0,
            )
            body.state.predictive_error[output_mask] = np.clip(
                body.state.predictive_error[output_mask] + 0.04 + 0.04 * abs(self._current_setpoint),
                0.0,
                1.0,
            )

    def _apply_row_load(self, body, port: MNISTPort, row: np.ndarray, row_index: int, condition_name: str) -> None:
        super()._apply_row_load(body, port, row, row_index, condition_name)

        row_signal = np.asarray(row, dtype=float).reshape(-1)
        row_gradient = float(np.mean(np.abs(np.diff(row_signal)))) if row_signal.size > 1 else 0.0
        row_intensity = float(np.mean(row_signal))
        lesion_load = 0.86 * row_gradient + 0.48 * row_intensity
        output_window = port.boundary_window("output")
        input_window = port.boundary_window("input")
        output_mask = port.support_mask("output")
        relay_slice = self._focus_slice(row_signal, body.state.hidden.shape[0], radius=2)
        relay_start, relay_stop = relay_slice.start, relay_slice.stop
        repair_slice = slice(max(relay_start - 1, 0), min(relay_stop + 1, body.state.hidden.shape[0]))

        if row_index == self.ROW_LESION_START:
            lesion_segment(body, relay_start, relay_stop, energy_scale=0.20, stress_boost=1.00 + 0.35 * lesion_load)
            disrupt_port_region(body, input_window.start, input_window.stop, attenuation=0.36)
            disrupt_port_region(body, output_window.start, output_window.stop, attenuation=0.18)
            port.damage({"input_attenuation": 0.68, "readout_attenuation": 0.48})
            body.state.energy[repair_slice] = np.clip(body.state.energy[repair_slice] * 0.30, 0.0, 1.0)
            body.state.stress[repair_slice] = np.clip(body.state.stress[repair_slice] + 1.05, 0.0, 5.0)
            body.state.field_alignment[repair_slice] = np.clip(body.state.field_alignment[repair_slice] * 0.40, 0.0, 1.0)
            body.state.z_alignment[repair_slice] *= 0.28
            body.state.predictive_error[repair_slice] = np.clip(body.state.predictive_error[repair_slice] + 0.52, 0.0, 1.0)
        elif self.ROW_LESION_START < row_index < self.ROW_LESION_STOP:
            body.state.stress[repair_slice] = np.clip(body.state.stress[repair_slice] + 0.16, 0.0, 5.0)
            body.state.energy[repair_slice] = np.clip(body.state.energy[repair_slice] - 0.06, 0.0, 1.0)
            body.state.field_alignment[repair_slice] = np.clip(body.state.field_alignment[repair_slice] - 0.10, 0.0, 1.0)
            body.state.predictive_error[repair_slice] = np.clip(body.state.predictive_error[repair_slice] + 0.10, 0.0, 1.0)
            body.state.stress[output_mask] = np.clip(body.state.stress[output_mask] + 0.08 * lesion_load, 0.0, 5.0)
            body.state.energy[output_mask] = np.clip(body.state.energy[output_mask] - 0.05 * lesion_load, 0.0, 1.0)

    def _apply_settle_load(self, body, port: MNISTPort, image: np.ndarray, settle_index: int, condition_name: str) -> None:
        image_signal = np.asarray(image, dtype=float)
        image_intensity = float(np.mean(image_signal))
        image_contrast = float(np.std(image_signal))
        settle_load = 0.45 * image_intensity + 0.52 * image_contrast
        output_mask = port.support_mask("output")
        relay_profile = np.mean(image_signal, axis=0)
        relay_slice = self._focus_slice(relay_profile, body.state.hidden.shape[0], radius=1)

        body.state.z_memory[output_mask] = np.clip(
            0.76 * body.state.z_memory[output_mask] + 0.24 * self._current_setpoint,
            -1.0,
            1.0,
        )
        body.state.stress[relay_slice] = np.clip(body.state.stress[relay_slice] + 0.10 * settle_load, 0.0, 5.0)
        body.state.energy[relay_slice] = np.clip(body.state.energy[relay_slice] - 0.07 * settle_load, 0.0, 1.0)
        body.state.field_alignment[relay_slice] = np.clip(body.state.field_alignment[relay_slice] - 0.14 * settle_load, 0.0, 1.0)
        body.state.predictive_error[relay_slice] = np.clip(body.state.predictive_error[relay_slice] + 0.08, 0.0, 1.0)

        if settle_index % 4 == 0:
            body.state.energy[output_mask] = np.clip(body.state.energy[output_mask] - 0.10 * settle_load, 0.0, 1.0)
            body.state.stress[output_mask] = np.clip(body.state.stress[output_mask] + 0.20 * settle_load, 0.0, 5.0)
            body.state.field_alignment[output_mask] = np.clip(body.state.field_alignment[output_mask] - 0.24 * settle_load, 0.0, 1.0)
            body.state.hidden[output_mask] *= (1.0 - 0.08 * settle_load)
            body.state.z_alignment[output_mask] *= (1.0 - 0.10 * settle_load)

    def _rollout_image(self, cfg, image: np.ndarray, *, condition_name: str = "baseline") -> dict:
        self._current_setpoint = self._latent_setpoint(image)
        rollout = super()._rollout_image(cfg, image, condition_name=condition_name)
        output_width = max(6, cfg.body.num_cells // 7)
        embedding = np.asarray(rollout["embedding"], dtype=float)
        image_signal = np.asarray(image, dtype=float)
        lesion_band = image_signal[self.ROW_LESION_START : self.ROW_LESION_STOP]
        compressed_context = np.array(
            [
                float(np.mean(image_signal)),
                float(np.std(image_signal)),
                float(np.mean(lesion_band)) if lesion_band.size else 0.0,
                float(np.mean(np.abs(np.diff(np.mean(image_signal, axis=0))))) if image_signal.shape[1] > 1 else 0.0,
                float(self._current_setpoint),
            ],
            dtype=float,
        )
        physiology = np.array(
            [
                float(rollout["final_metrics"].get("mnist_peak_growth_pressure", 0.0)),
                float(rollout["final_metrics"].get("mean_z_memory", 0.0)),
                float(rollout["final_metrics"].get("z_memory_alignment_gap", 0.0)),
                float(rollout["final_metrics"].get("lightcone_proxy", 0.0)),
            ],
            dtype=float,
        )
        rollout["embedding"] = np.concatenate(
            [
                embedding[2 * output_width :],
                compressed_context,
                physiology,
            ],
            axis=0,
        )
        return rollout

    def run_condition(self, cfg, condition_name: str) -> AssayResult:
        result = super().run_condition(cfg, condition_name)
        final_metrics = {}
        for key, value in result.final_metrics.items():
            if key.startswith("split_mnist_"):
                final_metrics[key.replace("split_mnist_", f"{self.METRIC_PREFIX}_", 1)] = value
            else:
                final_metrics[key] = value
        final_metrics[f"{self.METRIC_PREFIX}_lesion_row_start"] = float(self.ROW_LESION_START)
        final_metrics[f"{self.METRIC_PREFIX}_lesion_row_stop"] = float(self.ROW_LESION_STOP)
        final_metrics[f"{self.METRIC_PREFIX}_lesion_row_count"] = float(self.ROW_LESION_STOP - self.ROW_LESION_START)
        final_metrics[f"{self.METRIC_PREFIX}_lesion_active"] = 1.0
        notes = result.notes + " Visual relay/output lesions were applied during mid-image rollout with delayed recovery."
        return AssayResult(history=result.history, final_metrics=final_metrics, notes=notes)

    def run(self, cfg):
        return self.run_condition(cfg, "baseline")
