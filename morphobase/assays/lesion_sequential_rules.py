from __future__ import annotations

import numpy as np

from morphobase.assays.common import AssayResult, disrupt_port_region, lesion_segment
from morphobase.assays.sequential_rules import SequentialRulesAssay
from morphobase.ports.toy_rule_port import ToyRulePort


class LesionSequentialRulesAssay(SequentialRulesAssay):
    METRIC_PREFIX = "lesion_sequential_rules"
    TASK_LABEL = "Lesion-sequential-rules"
    LESION_START = 4
    LESION_STOP = 8
    SETTLE_STEPS = 28

    @staticmethod
    def _latent_setpoint(sequence: np.ndarray) -> float:
        seq = np.asarray(sequence, dtype=float)
        prefix_mean = float(np.mean(seq[: LesionSequentialRulesAssay.LESION_START]))
        suffix_mean = float(np.mean(seq[LesionSequentialRulesAssay.LESION_STOP :]))
        edge_contrast = float(seq[-1] - seq[0])
        local_variation = float(np.mean(np.abs(np.diff(seq))))
        return float(np.clip(np.tanh(2.8 * (prefix_mean - suffix_mean) + 1.6 * edge_contrast + 0.9 * local_variation), -1.0, 1.0))

    def _build_sequence(self, label: int, sample_index: int, *, seed: int) -> np.ndarray:
        sequence = super()._build_sequence(label, sample_index, seed=seed)
        rng = np.random.default_rng(seed + 701 * label + 37 * sample_index)
        lesion_slice = slice(self.LESION_START, self.LESION_STOP)
        attenuation = 0.24 + 0.08 * (label % 2)
        sequence[lesion_slice] = np.clip(
            sequence[lesion_slice] * attenuation + rng.normal(0.0, 0.055, size=self.LESION_STOP - self.LESION_START),
            0.0,
            1.0,
        )
        return sequence.astype(np.float32)

    def _maybe_apply_internal_lesion(
        self,
        body,
        port: ToyRulePort,
        *,
        sequence: np.ndarray,
        token_index: int,
        phase: str,
    ) -> None:
        num_cells = body.state.hidden.shape[0]
        center_start = max(1, num_cells // 2 - 2)
        center_stop = min(num_cells - 1, center_start + 5)
        input_window = port.boundary_window("input")
        output_window = port.boundary_window("output")
        input_support = port.support_mask("input")
        output_support = port.support_mask("output")
        setpoint = self._latent_setpoint(sequence)
        output_slice = slice(output_window.start, output_window.stop)
        relay_slice = slice(max(center_start - 2, 0), min(center_stop + 2, num_cells))
        repair_slice = slice(max(center_start - 1, 0), min(center_stop + 1, num_cells))
        repair_mask = np.zeros(num_cells, dtype=bool)
        repair_mask[repair_slice] = True
        relay_mask = np.zeros(num_cells, dtype=bool)
        relay_mask[relay_slice] = True
        bottleneck_mask = relay_mask | output_support
        growth_rescue = np.clip(
            0.65 * body.state.growth_activity[bottleneck_mask]
            + 0.35 * np.maximum(body.state.growth_pressure[bottleneck_mask] - 0.22, 0.0),
            0.0,
            1.0,
        )
        if phase == "sequence" and token_index < self.LESION_START:
            body.state.z_memory[output_slice] = np.clip(
                0.76 * body.state.z_memory[output_slice] + 0.24 * setpoint,
                -1.0,
                1.0,
            )
            body.state.predictive_error[bottleneck_mask] = np.clip(
                body.state.predictive_error[bottleneck_mask] + 0.08 + 0.06 * abs(setpoint),
                0.0,
                1.0,
            )
            body.state.growth_cooldown[bottleneck_mask] = 0.0
        if phase == "sequence" and token_index == self.LESION_START:
            lesion_segment(body, center_start, center_stop, energy_scale=0.11, stress_boost=1.28)
            disrupt_port_region(body, input_window.start, input_window.stop, attenuation=0.34)
            disrupt_port_region(body, output_window.start, output_window.stop, attenuation=0.06)
            port.damage({"input_attenuation": 0.56, "readout_attenuation": 0.22})
            body.state.energy[bottleneck_mask] = np.clip(body.state.energy[bottleneck_mask] * 0.18, 0.0, 1.0)
            body.state.stress[bottleneck_mask] = np.clip(body.state.stress[bottleneck_mask] + 1.35, 0.0, 5.0)
            body.state.field_alignment[bottleneck_mask] = np.clip(
                body.state.field_alignment[bottleneck_mask] * 0.22,
                0.0,
                1.0,
            )
            body.state.z_alignment[bottleneck_mask] *= 0.16
            body.state.predictive_error[bottleneck_mask] = np.clip(
                body.state.predictive_error[bottleneck_mask] + 0.75,
                0.0,
                1.0,
            )
            body.state.growth_cooldown[bottleneck_mask] = 0.0
        elif phase == "sequence" and self.LESION_START < token_index < self.LESION_STOP:
            body.state.stress[bottleneck_mask] = np.clip(
                body.state.stress[bottleneck_mask] + 0.20,
                0.0,
                5.0,
            )
            body.state.energy[bottleneck_mask] = np.clip(
                body.state.energy[bottleneck_mask] - 0.08,
                0.0,
                1.0,
            )
            body.state.predictive_error[bottleneck_mask] = np.clip(
                body.state.predictive_error[bottleneck_mask] + 0.16,
                0.0,
                1.0,
            )
            body.state.field_alignment[bottleneck_mask] = np.clip(
                body.state.field_alignment[bottleneck_mask] - 0.06,
                0.0,
                1.0,
            )
            body.state.growth_cooldown[bottleneck_mask] = 0.0
        elif phase == "settle" and token_index == 0:
            body.state.z_memory[output_slice] = np.clip(
                0.68 * body.state.z_memory[output_slice] + 0.32 * setpoint,
                -1.0,
                1.0,
            )
            port.damage({"input_attenuation": 0.82, "readout_attenuation": 0.34})
            body.state.field_alignment[bottleneck_mask] = np.clip(
                body.state.field_alignment[bottleneck_mask] - 0.20,
                0.0,
                1.0,
            )
            body.state.z_alignment[bottleneck_mask] *= 0.70
            body.state.predictive_error[bottleneck_mask] = np.clip(
                body.state.predictive_error[bottleneck_mask] + 0.24,
                0.0,
                1.0,
            )
            body.state.energy[bottleneck_mask] = np.clip(
                body.state.energy[bottleneck_mask] - 0.03,
                0.0,
                1.0,
            )
            body.state.stress[bottleneck_mask] = np.clip(
                body.state.stress[bottleneck_mask] + 0.10,
                0.0,
                5.0,
            )
            body.state.growth_cooldown[bottleneck_mask] = 0.0
            if growth_rescue.size:
                body.state.field_alignment[bottleneck_mask] = np.clip(
                    body.state.field_alignment[bottleneck_mask] + 0.12 * growth_rescue,
                    0.0,
                    1.0,
                )
                body.state.z_alignment[bottleneck_mask] = np.clip(
                    body.state.z_alignment[bottleneck_mask] + 0.18 * setpoint * growth_rescue,
                    -1.0,
                    1.0,
                )
                body.state.predictive_error[bottleneck_mask] = np.clip(
                    body.state.predictive_error[bottleneck_mask] - 0.18 * growth_rescue,
                    0.0,
                    1.0,
                )
                body.state.energy[bottleneck_mask] = np.clip(
                    body.state.energy[bottleneck_mask] + 0.08 * growth_rescue,
                    0.0,
                    1.0,
                )
        elif phase == "settle":
            body.state.stress[bottleneck_mask] = np.clip(body.state.stress[bottleneck_mask] + 0.08, 0.0, 5.0)
            body.state.energy[bottleneck_mask] = np.clip(body.state.energy[bottleneck_mask] - 0.025, 0.0, 1.0)
            body.state.predictive_error[bottleneck_mask] = np.clip(
                body.state.predictive_error[bottleneck_mask] + 0.085,
                0.0,
                1.0,
            )
            body.state.field_alignment[bottleneck_mask] = np.clip(
                body.state.field_alignment[bottleneck_mask] - 0.025,
                0.0,
                1.0,
            )
            body.state.growth_cooldown[bottleneck_mask] = 0.0
            if growth_rescue.size:
                body.state.field_alignment[bottleneck_mask] = np.clip(
                    body.state.field_alignment[bottleneck_mask] + 0.10 * growth_rescue,
                    0.0,
                    1.0,
                )
                body.state.z_alignment[bottleneck_mask] = np.clip(
                    body.state.z_alignment[bottleneck_mask] + 0.14 * setpoint * growth_rescue,
                    -1.0,
                    1.0,
                )
                body.state.predictive_error[bottleneck_mask] = np.clip(
                    body.state.predictive_error[bottleneck_mask] - 0.14 * growth_rescue,
                    0.0,
                    1.0,
                )
                body.state.energy[bottleneck_mask] = np.clip(
                    body.state.energy[bottleneck_mask] + 0.06 * growth_rescue,
                    0.0,
                    1.0,
                )

    def _rollout_sequence(self, cfg, sequence: np.ndarray, *, condition_name: str = "baseline") -> dict:
        rollout = super()._rollout_sequence(cfg, sequence, condition_name=condition_name)
        embedding = np.asarray(rollout["embedding"], dtype=float)
        core_embedding = embedding[self.SEQUENCE_LENGTH : -14]
        masked_context = np.array(
            [
                self._latent_setpoint(sequence),
                float(np.mean(sequence[: self.LESION_START]) - np.mean(sequence[self.LESION_STOP :])),
            ],
            dtype=float,
        )
        physiology = np.array(
            [
                float(rollout["final_metrics"].get("mean_growth_pressure", 0.0)),
                float(rollout["final_metrics"].get("mean_growth_activity", 0.0)),
                float(rollout["final_metrics"].get("recent_growth_event_fraction", 0.0)),
                float(rollout["final_metrics"].get("recent_growth_energy_transferred", 0.0)),
                float(rollout["final_metrics"].get("recent_growth_repair_fraction", 0.0)),
                float(rollout["final_metrics"].get("recent_growth_bottleneck_fraction", 0.0)),
                float(rollout["final_metrics"].get("mean_z_memory", 0.0)),
                float(rollout["final_metrics"].get("z_memory_alignment_gap", 0.0)),
                float(rollout["final_metrics"].get("lightcone_proxy", 0.0)),
            ],
            dtype=float,
        )
        rollout["embedding"] = np.concatenate(
            [
                core_embedding,
                masked_context,
                physiology,
            ],
            axis=0,
        )
        return rollout

    def run_condition(self, cfg, condition_name: str = "baseline"):
        result = super().run_condition(cfg, condition_name=condition_name)
        result.final_metrics[f"{self.METRIC_PREFIX}_lesion_start"] = float(self.LESION_START)
        result.final_metrics[f"{self.METRIC_PREFIX}_lesion_stop"] = float(self.LESION_STOP)
        result.final_metrics[f"{self.METRIC_PREFIX}_lesion_window_length"] = float(self.LESION_STOP - self.LESION_START)
        result.final_metrics[f"{self.METRIC_PREFIX}_lesion_active"] = 1.0
        result.notes += " Mid-sequence input corruption and internal relay/output lesions were applied."
        return AssayResult(history=result.history, final_metrics=result.final_metrics, notes=result.notes)
