from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from morphobase.assays.common import AssayResult, AssayRunner, build_synthetic_body, disrupt_port_region, lesion_segment
from morphobase.diagnostics.metrics import lightcone_proxy, summarize_state
from morphobase.organism.scheduler import Scheduler
from morphobase.ports.control_port import ControlPort
from morphobase.training.trainer import SequentialLinearTrainer


class GridworldRemapAssay(AssayRunner):
    METRIC_PREFIX = "gridworld_remap"
    TASK_LABEL = "Gridworld remap"
    GRID_HEIGHT = 5
    GRID_WIDTH = 5
    ACTION_LABELS = np.array([0, 1, 2, 3, 4], dtype=int)  # up, left, stay, right, down
    ACTION_DELTAS = {
        0: (-1, 0),
        1: (0, -1),
        2: (0, 0),
        3: (0, 1),
        4: (1, 0),
    }
    TASK_SPECS = (
        {
            "name": "north_east_band",
            "reward_zone": ((4, 3), (4, 4)),
            "obstacles": ((2, 2),),
            "hazards": ((1, 3), (2, 3)),
            "obs_remap": "identity",
            "port_remap": {},
        },
        {
            "name": "south_west_band",
            "reward_zone": ((0, 0), (0, 1)),
            "obstacles": ((2, 1), (2, 2)),
            "hazards": ((3, 1), (3, 2)),
            "obs_remap": "identity",
            "port_remap": {"input_shift": 1, "output_shift": 1},
        },
        {
            "name": "mirrored_east",
            "reward_zone": ((3, 4), (4, 4)),
            "obstacles": ((1, 2), (2, 2)),
            "hazards": ((3, 3), (4, 2)),
            "obs_remap": "mirror_x",
            "port_remap": {"flip": True},
        },
        {
            "name": "upper_shifted",
            "reward_zone": ((2, 4), (3, 4)),
            "obstacles": ((1, 1), (2, 1), (3, 1)),
            "hazards": ((3, 2), (4, 2)),
            "obs_remap": "rotate90",
            "port_remap": {"input_shift": 2, "output_shift": 1, "scale": 0.92},
        },
        {
            "name": "swap_axes_gate",
            "reward_zone": ((2, 4), (3, 4)),
            "obstacles": ((2, 3),),
            "hazards": ((1, 2),),
            "obs_remap": "transpose",
            "port_remap": {"input_shift": 1, "output_shift": 1, "scale": 0.98},
        },
    )
    SUPPORT_EPISODES = 18
    EVAL_EPISODES = 12
    SUPPORT_CURRICULUM_SEED = 31415
    EVAL_CURRICULUM_SEED = 27182
    HORIZON = 12
    SETTLE_STEPS = 1
    REPAIR_PROBE_SETTLE_STEPS = 9
    PATCH_RADIUS = 1
    ABLATION_CONDITIONS = ("baseline", "no_growth", "no_stress", "no_z_field")
    GROWTH_TRIGGER_THRESHOLD = 0.18
    CONDITION_SPECS = {
        "baseline": {"allow_growth": True, "z_field": True},
        "no_growth": {"allow_growth": False, "z_field": True},
        "no_stress": {"allow_growth": True, "z_field": True},
        "no_z_field": {"allow_growth": True, "z_field": False},
    }

    def __init__(self) -> None:
        self.challenge_variant = "standard"

    @staticmethod
    def _inside(position: tuple[int, int]) -> bool:
        row, col = position
        return 0 <= row < GridworldRemapAssay.GRID_HEIGHT and 0 <= col < GridworldRemapAssay.GRID_WIDTH

    @staticmethod
    def _positions(spec: dict, key: str) -> set[tuple[int, int]]:
        return {tuple(int(v) for v in position) for position in spec[key]}

    @staticmethod
    def _one_hot(index: int, size: int) -> np.ndarray:
        vec = np.zeros(size, dtype=float)
        vec[int(index)] = 1.0
        return vec

    def _initial_position(self, task_spec: dict, *, episode_seed: int) -> tuple[int, int]:
        rng = np.random.default_rng(episode_seed)
        forbidden = self._positions(task_spec, "reward_zone") | self._positions(task_spec, "obstacles")
        candidates = [
            (row, col)
            for row in range(self.GRID_HEIGHT)
            for col in range(self.GRID_WIDTH)
            if (row, col) not in forbidden
        ]
        return candidates[int(rng.integers(0, len(candidates)))]

    def _grid_map(self, positions: Iterable[tuple[int, int]]) -> np.ndarray:
        grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=float)
        for row, col in positions:
            grid[int(row), int(col)] = 1.0
        return grid

    @staticmethod
    def _action_setpoint(task_spec: dict, target_action: int) -> float:
        reward_positions = np.asarray(task_spec["reward_zone"], dtype=float)
        reward_bias = float(np.mean(reward_positions[:, 0]) - np.mean(reward_positions[:, 1])) / 4.0
        action_bias = (float(target_action) - 2.0) / 2.0
        return float(np.clip(np.tanh(1.6 * action_bias + 1.0 * reward_bias), -1.0, 1.0))

    def _transform_patch(self, patch: np.ndarray, remap: str) -> np.ndarray:
        if remap == "identity":
            return patch
        if remap == "mirror_x":
            return patch[:, ::-1]
        if remap == "mirror_y":
            return patch[::-1, :]
        if remap == "rotate90":
            return np.rot90(patch, k=1)
        if remap == "transpose":
            return patch.T
        raise ValueError(f"Unknown remap: {remap}")

    def _transform_vector(self, dy: float, dx: float, remap: str) -> tuple[float, float]:
        if remap == "identity":
            return dy, dx
        if remap == "mirror_x":
            return dy, -dx
        if remap == "mirror_y":
            return -dy, dx
        if remap == "rotate90":
            return -dx, dy
        if remap == "transpose":
            return dx, dy
        raise ValueError(f"Unknown remap: {remap}")

    def _local_patch(self, grid: np.ndarray, position: tuple[int, int]) -> np.ndarray:
        row, col = position
        radius = self.PATCH_RADIUS
        patch = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=float)
        for patch_row, source_row in enumerate(range(row - radius, row + radius + 1)):
            for patch_col, source_col in enumerate(range(col - radius, col + radius + 1)):
                if 0 <= source_row < self.GRID_HEIGHT and 0 <= source_col < self.GRID_WIDTH:
                    patch[patch_row, patch_col] = grid[source_row, source_col]
        return patch

    def _observation(
        self,
        task_spec: dict,
        position: tuple[int, int],
        prev_action: int,
    ) -> np.ndarray:
        reward_grid = self._grid_map(self._positions(task_spec, "reward_zone"))
        obstacle_grid = self._grid_map(self._positions(task_spec, "obstacles"))
        hazard_grid = self._grid_map(self._positions(task_spec, "hazards"))

        reward_patch = self._transform_patch(self._local_patch(reward_grid, position), str(task_spec["obs_remap"]))
        obstacle_patch = self._transform_patch(self._local_patch(obstacle_grid, position), str(task_spec["obs_remap"]))
        hazard_patch = self._transform_patch(self._local_patch(hazard_grid, position), str(task_spec["obs_remap"]))

        reward_center = np.mean(np.array(list(self._positions(task_spec, "reward_zone")), dtype=float), axis=0)
        dy = float((reward_center[0] - position[0]) / max(self.GRID_HEIGHT - 1, 1))
        dx = float((reward_center[1] - position[1]) / max(self.GRID_WIDTH - 1, 1))
        dy, dx = self._transform_vector(dy, dx, str(task_spec["obs_remap"]))
        distance = float(abs(dy) + abs(dx))

        port_shift = float(task_spec["port_remap"].get("input_shift", 0)) / 3.0
        output_shift = float(task_spec["port_remap"].get("output_shift", 0)) / 3.0
        scale = float(task_spec["port_remap"].get("scale", 1.0))
        flip = 1.0 if bool(task_spec["port_remap"].get("flip", False)) else 0.0

        observation = np.concatenate(
            [
                reward_patch.reshape(-1),
                obstacle_patch.reshape(-1),
                hazard_patch.reshape(-1),
                np.array([dy, dx, distance, scale, flip, port_shift, output_shift], dtype=float),
                self._one_hot(prev_action, len(self.ACTION_LABELS)),
            ],
            axis=0,
        )
        if self.challenge_variant == "repair_probe":
            obs = np.asarray(observation, dtype=np.float32).copy()
            lesion_strength = 0.24 + 0.06 * ((position[0] + position[1] + prev_action) % 3)
            obs[9:18] *= (1.0 - lesion_strength)
            obs[27:34] = np.clip(obs[27:34] - 0.16 * lesion_strength, -1.0, 1.0)
            obs[34:39] *= (1.0 - 0.26 * lesion_strength)
            return obs
        return observation

    def _transition(self, task_spec: dict, position: tuple[int, int], action: int) -> tuple[int, int]:
        delta = self.ACTION_DELTAS[int(action)]
        candidate = (position[0] + delta[0], position[1] + delta[1])
        obstacles = self._positions(task_spec, "obstacles")
        if not self._inside(candidate) or candidate in obstacles:
            return position
        return candidate

    def _state_cost(self, task_spec: dict, position: tuple[int, int]) -> float:
        reward_zone = self._positions(task_spec, "reward_zone")
        if position in reward_zone:
            return 0.0
        reward_center = np.mean(np.array(list(reward_zone), dtype=float), axis=0)
        hazard_penalty = 1.8 if position in self._positions(task_spec, "hazards") else 0.0
        return float(abs(position[0] - reward_center[0]) + abs(position[1] - reward_center[1]) + hazard_penalty)

    def _expert_action(self, task_spec: dict, position: tuple[int, int], prev_action: int) -> int:
        reward_zone = self._positions(task_spec, "reward_zone")
        if position in reward_zone:
            return 2

        best_action = 2
        best_score = float("inf")
        for action in self.ACTION_LABELS:
            next_position = self._transition(task_spec, position, int(action))
            action_cost = 0.08 if int(action) == prev_action else 0.0
            stay_penalty = 0.4 if int(action) == 2 and position not in reward_zone else 0.0
            score = self._state_cost(task_spec, next_position) + action_cost + stay_penalty
            if score < best_score - 1e-8 or (abs(score - best_score) <= 1e-8 and abs(int(action) - 2) > abs(best_action - 2)):
                best_score = score
                best_action = int(action)
        return best_action

    @staticmethod
    def _apply_control_load(body, port: ControlPort, observation: np.ndarray) -> None:
        support_mask = port.support_mask("input")
        boundary_mask = port.boundary_mask("input")
        signal = np.asarray(observation, dtype=float)
        challenge = float(np.mean(signal[: 3 * 3 * 3]))
        mismatch = float(np.mean(np.abs(signal[-len(GridworldRemapAssay.ACTION_LABELS) :] - 0.2)))
        body.state.stress[support_mask] = np.clip(
            body.state.stress[support_mask] + 0.010 * challenge + 0.008 * mismatch,
            0.0,
            5.0,
        )
        body.state.energy[support_mask] = np.clip(
            body.state.energy[support_mask] - 0.005 * challenge,
            0.0,
            1.0,
        )
        body.state.field_alignment[support_mask] = np.clip(
            body.state.field_alignment[support_mask] + 0.012 * challenge - 0.007 * mismatch,
            0.0,
            1.0,
        )
        body.state.z_alignment[boundary_mask] = np.clip(
            body.state.z_alignment[boundary_mask] + 0.020 * signal[27:36].mean(),
            -1.0,
            1.0,
        )

    def _apply_probe_load(self, body, port: ControlPort, observation: np.ndarray, target_action: int, phase_index: int) -> None:
        if self.challenge_variant != "repair_probe":
            return
        signal = np.asarray(observation, dtype=float)
        challenge = float(np.mean(signal[:27]))
        hazard_drive = float(np.mean(signal[18:27]))
        directional_mismatch = float(np.mean(np.abs(signal[-len(self.ACTION_LABELS) :] - 0.2)))
        challenge_load = 0.46 * challenge + 0.34 * hazard_drive + 0.28 * directional_mismatch
        support_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        output_mask = port.support_mask("output")
        relay_slice = slice(max(0, body.state.hidden.shape[0] // 2 - 3), min(body.state.hidden.shape[0], body.state.hidden.shape[0] // 2 + 3))

        body.state.stress[support_mask] = np.clip(body.state.stress[support_mask] + 0.16 * challenge_load, 0.0, 5.0)
        body.state.energy[support_mask] = np.clip(body.state.energy[support_mask] - 0.08 * challenge_load, 0.0, 1.0)
        body.state.field_alignment[support_mask] = np.clip(
            body.state.field_alignment[support_mask] - 0.14 * challenge_load,
            0.0,
            1.0,
        )
        body.state.predictive_error[relay_slice] = np.clip(
            body.state.predictive_error[relay_slice] + 0.12 * challenge_load,
            0.0,
            1.0,
        )
        body.state.z_memory[output_mask] = np.clip(
            0.84 * body.state.z_memory[output_mask] + 0.16 * np.tanh(0.8 * (target_action - 2)),
            -1.0,
            1.0,
        )

        if phase_index % 2 == 0:
            body.state.hidden[output_mask] *= (1.0 - 0.10 * challenge_load)
            body.state.membrane[output_mask] = np.clip(
                body.state.membrane[output_mask] * (1.0 - 0.14 * challenge_load),
                -1.0,
                1.0,
            )
            body.state.stress[output_mask] = np.clip(body.state.stress[output_mask] + 0.20 * challenge_load, 0.0, 5.0)
            body.state.energy[output_mask] = np.clip(body.state.energy[output_mask] - 0.09 * challenge_load, 0.0, 1.0)
            body.state.field_alignment[output_mask] = np.clip(
                body.state.field_alignment[output_mask] - 0.18 * challenge_load,
                0.0,
                1.0,
            )
            body.state.z_alignment[output_mask] *= (1.0 - 0.10 * challenge_load)

    def _apply_settle_load(self, body, port: ControlPort, observation: np.ndarray, target_action: int, settle_index: int) -> None:
        if self.challenge_variant != "repair_probe":
            return
        signal = np.asarray(observation, dtype=float)
        settle_load = 0.30 * float(np.mean(signal[:9])) + 0.34 * float(np.mean(signal[18:27])) + 0.18 * abs(target_action - 2)
        relay_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        output_mask = port.support_mask("output")
        body.state.stress[relay_mask] = np.clip(body.state.stress[relay_mask] + 0.12 * settle_load, 0.0, 5.0)
        body.state.energy[relay_mask] = np.clip(body.state.energy[relay_mask] - 0.06 * settle_load, 0.0, 1.0)
        body.state.field_alignment[relay_mask] = np.clip(
            body.state.field_alignment[relay_mask] - 0.12 * settle_load,
            0.0,
            1.0,
        )
        if settle_index % 2 == 0:
            body.state.hidden[output_mask] *= (1.0 - 0.08 * settle_load)
            body.state.membrane[output_mask] = np.clip(
                body.state.membrane[output_mask] * (1.0 - 0.12 * settle_load),
                -1.0,
                1.0,
            )

    def _condition_spec(self, condition_name: str) -> dict[str, bool]:
        if condition_name not in self.CONDITION_SPECS:
            raise ValueError(f"Unknown GridworldRemap condition '{condition_name}'.")
        return dict(self.CONDITION_SPECS[condition_name])

    def _apply_condition_post_step(self, body, port: ControlPort, condition_name: str) -> None:
        if condition_name == "baseline":
            return
        support_mask = port.union_mask(port.support_mask("input"), port.support_mask("output"))
        distal_mask = port.distal_mask()
        if condition_name == "no_stress":
            body.state.stress[distal_mask] *= 0.08
            body.state.stress[support_mask] = np.clip(body.state.stress[support_mask] + 0.03, 0.0, 5.0)
            body.state.hidden[distal_mask] *= 0.986
            body.state.membrane[distal_mask] *= 0.96
            body.state.field_alignment[distal_mask] *= 0.84
            support_indices = np.flatnonzero(support_mask)
            distal_indices = np.flatnonzero(distal_mask)
            if support_indices.size and distal_indices.size:
                body.state.conductance[np.ix_(distal_indices, support_indices)] *= 0.90
                body.state.conductance[np.ix_(support_indices, distal_indices)] *= 0.90
                diagonal = np.diag_indices_from(body.state.conductance)
                body.state.conductance = np.clip(body.state.conductance, 0.0, 2.0)
                body.state.conductance[diagonal] = 1.0
        if condition_name == "no_z_field":
            body.state.z_alignment.fill(0.0)
            body.state.z_memory.fill(0.0)

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
        if self.challenge_variant != "repair_probe":
            return
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
                0.78 * body.state.z_memory[output_slice] + 0.22 * setpoint,
                -1.0,
                1.0,
            )
            body.state.predictive_error[repair_slice] = np.clip(
                body.state.predictive_error[repair_slice] + 0.06 + 0.04 * abs(setpoint),
                0.0,
                1.0,
            )
        if phase == "observation" and phase_index == 0:
            lesion_segment(body, relay_start, relay_stop, energy_scale=0.36, stress_boost=0.66 + 0.24 * lesion_strength)
            disrupt_port_region(body, input_window.start, input_window.stop, attenuation=0.18)
            disrupt_port_region(body, output_window.start, output_window.stop, attenuation=0.16)
            port.damage({"input_attenuation": 0.80, "readout_attenuation": 0.72})
            body.state.energy[repair_slice] = np.clip(body.state.energy[repair_slice] * 0.58, 0.0, 1.0)
            body.state.stress[repair_slice] = np.clip(body.state.stress[repair_slice] + 0.56, 0.0, 5.0)
            body.state.field_alignment[repair_slice] = np.clip(body.state.field_alignment[repair_slice] * 0.58, 0.0, 1.0)
            body.state.z_alignment[repair_slice] *= 0.46
            body.state.predictive_error[repair_slice] = np.clip(body.state.predictive_error[repair_slice] + 0.28, 0.0, 1.0)
        elif phase == "settle":
            body.state.z_memory[output_slice] = np.clip(
                0.72 * body.state.z_memory[output_slice] + 0.28 * setpoint,
                -1.0,
                1.0,
            )
            body.state.stress[output_slice] = np.clip(body.state.stress[output_slice] + 0.08, 0.0, 5.0)
            body.state.energy[repair_slice] = np.clip(body.state.energy[repair_slice] - 0.025, 0.0, 1.0)
            body.state.field_alignment[repair_slice] = np.clip(body.state.field_alignment[repair_slice] - 0.08, 0.0, 1.0)
            body.state.predictive_error[repair_slice] = np.clip(body.state.predictive_error[repair_slice] + 0.10, 0.0, 1.0)

    def _rollout_observation(
        self,
        cfg,
        task_spec: dict,
        observation: np.ndarray,
        target_action: int,
        *,
        condition_name: str = "baseline",
    ) -> dict:
        condition = self._condition_spec(condition_name)
        body = build_synthetic_body(cfg)
        port = ControlPort(cfg.body.num_cells)
        port.remap(task_spec["port_remap"])
        scheduler = Scheduler()
        history = []
        state_history = [body.state.copy()]
        z_history = [body.state.z_alignment.copy()]
        target_value = target_action / max(len(self.ACTION_LABELS) - 1, 1)

        port.apply_input(body, observation)
        self._apply_control_load(body, port, observation)
        self._apply_probe_load(body, port, observation, target_action, 0)
        self._maybe_apply_internal_lesion(
            body,
            port,
            task_spec=task_spec,
            observation=observation,
            target_action=target_action,
            phase="observation",
            phase_index=0,
        )
        self._apply_condition_post_step(body, port, condition_name)
        due = scheduler.due(body.state.step_count)
        body.step(
            due.fast,
            due.medium,
            due.slow,
            cfg.assay.noise_scale,
            target_value,
            allow_growth=condition["allow_growth"],
        )
        self._apply_condition_post_step(body, port, condition_name)
        z_history.append(body.state.z_alignment.copy())
        state_history.append(body.state.copy())
        history.append(summarize_state(body.state, z_history=z_history))

        settle_steps = self.REPAIR_PROBE_SETTLE_STEPS if self.challenge_variant == "repair_probe" else self.SETTLE_STEPS
        for settle_index in range(settle_steps):
            self._apply_settle_load(body, port, observation, target_action, settle_index)
            self._maybe_apply_internal_lesion(
                body,
                port,
                task_spec=task_spec,
                observation=observation,
                target_action=target_action,
                phase="settle",
                phase_index=settle_index,
            )
            self._apply_condition_post_step(body, port, condition_name)
            due = scheduler.due(body.state.step_count)
            body.step(
                due.fast,
                due.medium,
                due.slow,
                cfg.assay.noise_scale,
                target_value,
                allow_growth=condition["allow_growth"],
            )
            self._apply_condition_post_step(body, port, condition_name)
            z_history.append(body.state.z_alignment.copy())
            state_history.append(body.state.copy())
            history.append(summarize_state(body.state, z_history=z_history))

        final_metrics = summarize_state(body.state, z_history=z_history)
        final_metrics["lightcone_proxy"] = lightcone_proxy(state_history)
        boundary_state = port.capture_boundary_state(body, kind="output")
        hidden = np.asarray(boundary_state["hidden"], dtype=float)
        membrane = np.asarray(boundary_state["membrane"], dtype=float)
        field = np.asarray(boundary_state["field_alignment"], dtype=float)
        z_alignment = np.asarray(boundary_state["z_alignment"], dtype=float)
        z_memory = np.asarray(boundary_state["z_memory"], dtype=float)
        observation_vec = np.asarray(observation, dtype=float)
        embedding = np.concatenate(
            [
                observation_vec,
                hidden.reshape(-1),
                membrane.reshape(-1),
                field.reshape(-1),
                z_alignment.reshape(-1),
                z_memory.reshape(-1),
                np.array(
                    [
                        final_metrics["mean_energy"],
                        final_metrics["mean_stress"],
                        final_metrics["mean_plasticity"],
                        final_metrics["mean_z_alignment"],
                        final_metrics["mean_growth_pressure"],
                        final_metrics["lightcone_proxy"],
                    ],
                    dtype=float,
                ),
            ],
            axis=0,
        )
        if self.challenge_variant == "repair_probe":
            compressed_context = np.array(
                [
                    float(np.mean(observation_vec[0:9])),
                    float(np.mean(observation_vec[9:18])),
                    float(np.mean(observation_vec[18:27])),
                    float(np.linalg.norm(observation_vec[27:29])),
                    float(np.mean(observation_vec[29:34])),
                    float(observation_vec[34 + int(target_action)] if 34 + int(target_action) < observation_vec.size else 0.0),
                ],
                dtype=float,
            )
            physiology = np.array(
                [
                    float(final_metrics.get("mean_growth_pressure", 0.0)),
                    float(final_metrics.get("mean_z_memory", 0.0)),
                    float(final_metrics.get("z_memory_alignment_gap", 0.0)),
                    float(final_metrics.get("lightcone_proxy", 0.0)),
                ],
                dtype=float,
            )
            embedding = np.concatenate(
                [
                    embedding[observation_vec.size :],
                    compressed_context,
                    physiology,
                ],
                axis=0,
            )
        return {
            "embedding": embedding,
            "history": history,
            "final_metrics": final_metrics,
        }

    def _support_samples(self, task_spec: dict, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        observations = []
        labels = []
        rng = np.random.default_rng(seed)
        reward_zone = self._positions(task_spec, "reward_zone")
        for episode_index in range(self.SUPPORT_EPISODES):
            position = self._initial_position(task_spec, episode_seed=seed + 17 * episode_index)
            prev_action = int(rng.choice(self.ACTION_LABELS))
            for _ in range(2):
                if position in reward_zone:
                    position = self._initial_position(task_spec, episode_seed=seed + 701 + 19 * episode_index + _)
                    prev_action = int(rng.choice(self.ACTION_LABELS))
                observation = self._observation(task_spec, position, prev_action)
                action = self._expert_action(task_spec, position, prev_action)
                observations.append(observation)
                labels.append(action)
                position = self._transition(task_spec, position, action)
                prev_action = action
        return np.stack(observations, axis=0), np.asarray(labels, dtype=int)

    def _evaluate_episode(self, cfg, task_spec: dict, final_model, *, episode_seed: int, condition_name: str = "baseline") -> tuple[float, float]:
        position = self._initial_position(task_spec, episode_seed=episode_seed)
        prev_action = 2
        reward_zone = self._positions(task_spec, "reward_zone")
        for step in range(self.HORIZON):
            observation = self._observation(task_spec, position, prev_action)
            rollout = self._rollout_observation(
                cfg,
                task_spec,
                observation,
                self._expert_action(task_spec, position, prev_action),
                condition_name=condition_name,
            )
            action = int(final_model.predict(rollout["embedding"][None, :])[0])
            position = self._transition(task_spec, position, action)
            prev_action = action
            if position in reward_zone:
                efficiency = 1.0 - (step / max(self.HORIZON - 1, 1))
                return 1.0, float(np.clip(efficiency, 0.0, 1.0))
        return 0.0, 0.0

    def run_condition(self, cfg, condition_name: str = "baseline"):
        representative_history = None
        representative_metrics = None
        support_embeddings_by_task: list[np.ndarray] = []
        support_labels_by_task: list[np.ndarray] = []
        task_peak_scores: dict[int, float] = {}
        task_initial_scores: dict[int, float] = {}
        task_final_scores: dict[int, float] = {}
        task_efficiency_scores: dict[int, float] = {}
        evaluation_grid: list[list[float]] = []

        for task_index, task_spec in enumerate(self.TASK_SPECS):
            support_obs, support_labels = self._support_samples(task_spec, seed=self.SUPPORT_CURRICULUM_SEED + 43 * task_index)
            task_embeddings = []
            for obs, label in zip(support_obs, support_labels, strict=True):
                rollout = self._rollout_observation(cfg, task_spec, obs, int(label), condition_name=condition_name)
                task_embeddings.append(rollout["embedding"])
                if representative_history is None:
                    representative_history = rollout["history"]
                    representative_metrics = rollout["final_metrics"].copy()
            support_embeddings_by_task.append(np.stack(task_embeddings, axis=0))
            support_labels_by_task.append(support_labels)

        trainer = SequentialLinearTrainer(
            self.ACTION_LABELS.copy(),
            support_embeddings_by_task[0].shape[1],
            seed=cfg.run.seed + 719,
        )
        seen_train_embeddings: list[np.ndarray] = []
        seen_train_labels: list[np.ndarray] = []
        final_model = None

        for task_index, task_embeddings in enumerate(support_embeddings_by_task):
            seen_train_embeddings.append(task_embeddings)
            seen_train_labels.append(support_labels_by_task[task_index])
            train_embeddings = np.concatenate(seen_train_embeddings, axis=0)
            train_labels = np.concatenate(seen_train_labels, axis=0)
            final_model = trainer.train_task(train_embeddings, train_labels, epochs=80, learning_rate=0.12, l2=2e-4)

            row = []
            for seen_task_index in range(task_index + 1):
                successes = []
                efficiencies = []
                for episode_idx in range(self.EVAL_EPISODES):
                    success, efficiency = self._evaluate_episode(
                        cfg,
                        self.TASK_SPECS[seen_task_index],
                        final_model,
                        episode_seed=self.EVAL_CURRICULUM_SEED + 89 * seen_task_index + episode_idx,
                        condition_name=condition_name,
                    )
                    successes.append(success)
                    efficiencies.append(efficiency)
                success_rate = float(np.mean(successes))
                efficiency_rate = float(np.mean(efficiencies))
                row.append(success_rate)
                task_peak_scores[seen_task_index] = max(task_peak_scores.get(seen_task_index, 0.0), success_rate)
                if seen_task_index == task_index:
                    task_initial_scores[seen_task_index] = success_rate
                task_final_scores[seen_task_index] = success_rate
                task_efficiency_scores[seen_task_index] = efficiency_rate
            evaluation_grid.append(row)

        forgetting_values = [
            task_peak_scores[index] - task_final_scores.get(index, 0.0)
            for index in range(len(self.TASK_SPECS) - 1)
        ]
        bwt_values = [
            task_final_scores.get(index, 0.0) - task_initial_scores.get(index, 0.0)
            for index in range(len(self.TASK_SPECS) - 1)
        ]
        final_row = evaluation_grid[-1] if evaluation_grid else []
        final_success_mean = float(np.mean(final_row)) if final_row else 0.0
        peak_success_mean = float(np.mean(list(task_peak_scores.values()))) if task_peak_scores else 0.0
        mean_forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
        bwt = float(np.mean(bwt_values)) if bwt_values else 0.0
        mean_margin = final_model.mean_margin(np.concatenate(support_embeddings_by_task, axis=0)) if final_model is not None else 0.0
        efficiency_mean = float(np.mean(list(task_efficiency_scores.values()))) if task_efficiency_scores else 0.0

        final_metrics = representative_metrics or {}
        prefix = self.METRIC_PREFIX
        final_metrics[f"{prefix}_task_count"] = float(len(self.TASK_SPECS))
        final_metrics[f"{prefix}_final_success_mean"] = final_success_mean
        final_metrics[f"{prefix}_peak_success_mean"] = peak_success_mean
        final_metrics[f"{prefix}_mean_forgetting"] = mean_forgetting
        final_metrics[f"{prefix}_bwt"] = bwt
        final_metrics[f"{prefix}_mean_margin"] = float(mean_margin)
        final_metrics[f"{prefix}_efficiency_mean"] = efficiency_mean
        final_metrics[f"{prefix}_support_count"] = float(sum(len(labels) for labels in support_labels_by_task))
        final_metrics[f"{prefix}_eval_count"] = float(len(self.TASK_SPECS) * self.EVAL_EPISODES)
        final_metrics[f"{prefix}_grid_height"] = float(self.GRID_HEIGHT)
        final_metrics[f"{prefix}_grid_width"] = float(self.GRID_WIDTH)
        final_metrics[f"{prefix}_horizon"] = float(self.HORIZON)
        final_metrics[f"{prefix}_port_family_control"] = 1.0
        final_metrics[f"{prefix}_first_task_final_success"] = task_final_scores.get(0, 0.0)
        final_metrics[f"{prefix}_last_task_success"] = task_final_scores.get(len(self.TASK_SPECS) - 1, 0.0)
        final_metrics[f"{prefix}_condition_is_baseline"] = 1.0 if condition_name == "baseline" else 0.0
        final_metrics[f"{prefix}_condition_no_growth"] = 1.0 if condition_name == "no_growth" else 0.0
        final_metrics[f"{prefix}_condition_no_stress"] = 1.0 if condition_name == "no_stress" else 0.0
        final_metrics[f"{prefix}_condition_no_z_field"] = 1.0 if condition_name == "no_z_field" else 0.0
        final_metrics[f"{prefix}_peak_growth_pressure_mean"] = float(
            max(item.get("mean_growth_pressure", 0.0) for item in representative_history) if representative_history else final_metrics["mean_growth_pressure"]
        )
        final_metrics[f"{prefix}_growth_trigger_threshold"] = self.GROWTH_TRIGGER_THRESHOLD
        final_metrics[f"{prefix}_growth_trigger_crossed"] = 1.0 if final_metrics[f"{prefix}_peak_growth_pressure_mean"] >= self.GROWTH_TRIGGER_THRESHOLD else 0.0
        for task_index, success in task_final_scores.items():
            final_metrics[f"{prefix}_task_{task_index}_final_success"] = float(success)
            final_metrics[f"{prefix}_task_{task_index}_efficiency"] = float(task_efficiency_scores.get(task_index, 0.0))
        for task_index, peak_success in task_peak_scores.items():
            final_metrics[f"{prefix}_task_{task_index}_peak_success"] = float(peak_success)

        notes = (
            f"{self.TASK_LABEL} assay ran a tiny non-visual control bridge with shifted reward zones, hazards, and port remaps. "
            f"final_success={final_success_mean:.4f}; forgetting={mean_forgetting:.4f}; "
            f"bwt={bwt:.4f}; margin={mean_margin:.4f}; efficiency={efficiency_mean:.4f}; condition={condition_name}; challenge_variant={self.challenge_variant}."
        )
        return AssayResult(
            history=representative_history or [],
            final_metrics=final_metrics,
            notes=notes,
        )

    def run(self, cfg):
        return self.run_condition(cfg, "baseline")
