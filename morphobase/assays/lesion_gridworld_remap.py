from __future__ import annotations

import numpy as np

from morphobase.assays.common import AssayResult, disrupt_port_region, lesion_segment
from morphobase.assays.gridworld_remap import GridworldRemapAssay
from morphobase.ports.control_port import ControlPort


class LesionGridworldRemapAssay(GridworldRemapAssay):
    METRIC_PREFIX = "lesion_gridworld_remap"
    TASK_LABEL = "Lesion-gridworld-remap"
    SETTLE_STEPS = 33

    @staticmethod
    def _action_setpoint(task_spec: dict, target_action: int) -> float:
        reward_positions = np.asarray(task_spec["reward_zone"], dtype=float)
        reward_bias = float(np.mean(reward_positions[:, 0]) - np.mean(reward_positions[:, 1])) / 4.0
        action_bias = (float(target_action) - 2.0) / 2.0
        return float(np.clip(np.tanh(1.8 * action_bias + 1.2 * reward_bias), -1.0, 1.0))

    def _observation(
        self,
        task_spec: dict,
        position: tuple[int, int],
        prev_action: int,
    ) -> np.ndarray:
        observation = super()._observation(task_spec, position, prev_action)
        obs = np.asarray(observation, dtype=np.float32).copy()
        lesion_strength = 0.28 + 0.08 * ((position[0] + position[1] + prev_action) % 3)
        obs[9:18] *= (1.0 - lesion_strength)
        obs[27:34] = np.clip(obs[27:34] - 0.22 * lesion_strength, -1.0, 1.0)
        obs[34:39] *= (1.0 - 0.35 * lesion_strength)
        return obs

    def _maybe_apply_internal_lesion(
        self,
        body,
        port: ControlPort,
        *,
        task_spec: dict,
        observation: np.ndarray,
        target_action: int,
        phase: str,
        phase_index: int,
    ) -> None:
        input_window = port.boundary_window("input")
        output_window = port.boundary_window("output")
        relay_start = max(input_window.stop, body.state.hidden.shape[0] // 2 - 2)
        relay_stop = min(body.state.hidden.shape[0] - 1, relay_start + 4)
        repair_slice = slice(max(relay_start - 1, 0), min(relay_stop + 1, body.state.hidden.shape[0]))
        output_slice = slice(output_window.start, output_window.stop)
        lesion_strength = float(np.mean(np.abs(np.asarray(observation, dtype=float)[9:18])))
        setpoint = self._action_setpoint(task_spec, target_action)
        if phase == "observation":
            body.state.z_memory[output_slice] = np.clip(
                0.74 * body.state.z_memory[output_slice] + 0.26 * setpoint,
                -1.0,
                1.0,
            )
            body.state.predictive_error[repair_slice] = np.clip(
                body.state.predictive_error[repair_slice] + 0.08 + 0.05 * abs(setpoint),
                0.0,
                1.0,
            )
        if phase == "observation" and phase_index == 0:
            lesion_segment(body, relay_start, relay_stop, energy_scale=0.20, stress_boost=1.00 + 0.35 * lesion_strength)
            disrupt_port_region(body, input_window.start, input_window.stop, attenuation=0.24)
            disrupt_port_region(body, output_window.start, output_window.stop, attenuation=0.18)
            port.damage({"input_attenuation": 0.66, "readout_attenuation": 0.50})
            body.state.energy[repair_slice] = np.clip(body.state.energy[repair_slice] * 0.34, 0.0, 1.0)
            body.state.stress[repair_slice] = np.clip(body.state.stress[repair_slice] + 1.00, 0.0, 5.0)
            body.state.field_alignment[repair_slice] = np.clip(body.state.field_alignment[repair_slice] * 0.45, 0.0, 1.0)
            body.state.z_alignment[repair_slice] *= 0.32
            body.state.predictive_error[repair_slice] = np.clip(body.state.predictive_error[repair_slice] + 0.48, 0.0, 1.0)
        elif phase == "settle":
            body.state.z_memory[output_slice] = np.clip(
                0.70 * body.state.z_memory[output_slice] + 0.30 * setpoint,
                -1.0,
                1.0,
            )
            body.state.stress[output_window.start:output_window.stop] = np.clip(
                body.state.stress[output_window.start:output_window.stop] + 0.12,
                0.0,
                5.0,
            )
            body.state.energy[repair_slice] = np.clip(
                body.state.energy[repair_slice] - 0.03,
                0.0,
                1.0,
            )
            body.state.field_alignment[repair_slice] = np.clip(
                body.state.field_alignment[repair_slice] - 0.10,
                0.0,
                1.0,
            )
            body.state.predictive_error[repair_slice] = np.clip(
                body.state.predictive_error[repair_slice] + 0.14,
                0.0,
                1.0,
            )

    def _rollout_observation(
        self,
        cfg,
        task_spec: dict,
        observation: np.ndarray,
        target_action: int,
        *,
        condition_name: str = "baseline",
    ) -> dict:
        rollout = super()._rollout_observation(
            cfg,
            task_spec,
            observation,
            target_action,
            condition_name=condition_name,
        )
        embedding = np.asarray(rollout["embedding"], dtype=float)
        obs = np.asarray(observation, dtype=float)
        compressed_context = np.array(
            [
                float(np.mean(obs[0:9])),
                float(np.mean(obs[9:18])),
                float(np.mean(obs[18:27])),
                float(np.linalg.norm(obs[27:29])),
                float(np.mean(obs[29:34])),
                float(obs[34 + int(target_action)] if 34 + int(target_action) < obs.size else 0.0),
            ],
            dtype=float,
        )
        physiology = np.array(
            [
                float(rollout["final_metrics"].get("mean_growth_pressure", 0.0)),
                float(rollout["final_metrics"].get("mean_z_memory", 0.0)),
                float(rollout["final_metrics"].get("z_memory_alignment_gap", 0.0)),
                float(rollout["final_metrics"].get("lightcone_proxy", 0.0)),
            ],
            dtype=float,
        )
        rollout["embedding"] = np.concatenate(
            [
                embedding[obs.size :],
                compressed_context,
                physiology,
            ],
            axis=0,
        )
        return rollout

    def run_condition(self, cfg, condition_name: str = "baseline"):
        result = super().run_condition(cfg, condition_name=condition_name)
        result.final_metrics[f"{self.METRIC_PREFIX}_lesion_active"] = 1.0
        result.final_metrics[f"{self.METRIC_PREFIX}_observation_patch_lesion"] = 1.0
        result.notes += " Observation patches and internal control-boundary relays were lesioned during rollout."
        return AssayResult(history=result.history, final_metrics=result.final_metrics, notes=result.notes)
